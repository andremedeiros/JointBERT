import os
import logging
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn.functional as F

from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
init_logger()
logger = logging.getLogger(__name__)

# Global variables to store loaded model and config
model = None
tokenizer = None
args = None
device = None
intent_label_lst = None
slot_label_lst = None
pad_token_label_id = None
db_path = "predictions.db"


class PredConfig:
    """Configuration for prediction"""
    def __init__(self, model_dir="./chatterlands_model", batch_size=1, no_cuda=False):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.no_cuda = no_cuda


def init_database():
    """Initialize SQLite database for logging predictions with schema migration"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with original schema if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            request_text TEXT NOT NULL,
            predicted_intent TEXT,
            predicted_slots TEXT,
            response_data TEXT,
            error TEXT,
            processing_time_ms REAL
        )
    ''')

    # Get existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {column[1] for column in cursor.fetchall()}

    # Add new columns if they don't exist (schema migration)
    if 'intent_confidence' not in existing_columns:
        logger.info("Adding intent_confidence column to database")
        cursor.execute('ALTER TABLE predictions ADD COLUMN intent_confidence REAL DEFAULT NULL')

    if 'overall_confidence' not in existing_columns:
        logger.info("Adding overall_confidence column to database")
        cursor.execute('ALTER TABLE predictions ADD COLUMN overall_confidence REAL DEFAULT NULL')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def log_prediction(request_text, response_data=None, error=None, processing_time_ms=None):
    """Log a prediction request and response to the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        timestamp = datetime.utcnow().isoformat()
        predicted_intent = None
        predicted_slots = None
        response_json = None
        intent_confidence = None
        overall_confidence = None

        if response_data:
            predicted_intent = response_data.get('intent')
            predicted_slots = json.dumps(response_data.get('slots'))
            response_json = json.dumps(response_data)
            intent_confidence = response_data.get('intent_confidence')
            overall_confidence = response_data.get('overall_confidence')

        cursor.execute('''
            INSERT INTO predictions
            (timestamp, request_text, predicted_intent, predicted_slots, response_data, error, processing_time_ms, intent_confidence, overall_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, request_text, predicted_intent, predicted_slots, response_json, error, processing_time_ms, intent_confidence, overall_confidence))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging prediction to database: {str(e)}")


def get_device(pred_config):
    """Get the best available device"""
    # Priority: MPS (Metal) > CUDA > CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and not pred_config.no_cuda:
        return "mps"
    elif torch.cuda.is_available() and not pred_config.no_cuda:
        return "cuda"
    else:
        return "cpu"


def load_model_and_config(model_dir="./chatterlands_model"):
    """Load model and configuration at startup"""
    global model, tokenizer, args, device, intent_label_lst, slot_label_lst, pad_token_label_id

    pred_config = PredConfig(model_dir=model_dir)

    # Load training args
    args = torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'), weights_only=False)

    # Get device
    device = get_device(pred_config)
    logger.info(f"Using device: {device}")

    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            args.model_dir,
            args=args,
            intent_label_lst=get_intent_labels(args),
            slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception as e:
        raise Exception(f"Some model files might be missing: {str(e)}")

    # Load tokenizer and labels
    tokenizer = load_tokenizer(args)
    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)
    pad_token_label_id = args.ignore_index

    logger.info("Model and configuration loaded successfully")


def convert_text_to_dataset(text, tokenizer, args, pad_token_label_id):
    """Convert input text to TensorDataset"""
    # Split text into words
    words = text.strip().split()

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    cls_token_segment_id = 0
    pad_token_segment_id = 0
    sequence_a_segment_id = 0
    mask_padding_with_zero = True

    tokens = []
    slot_label_mask = []

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[: (args.max_seq_len - special_tokens_count)]
        slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    token_type_ids = [sequence_a_segment_id] * len(tokens)
    slot_label_mask += [pad_token_label_id]

    # Add [CLS] token
    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids
    slot_label_mask = [pad_token_label_id] + slot_label_mask

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

    # Convert to tensors
    all_input_ids = torch.tensor([input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    all_token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    all_slot_label_mask = torch.tensor([slot_label_mask], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset, words


def parse_slots_to_entities(words, slot_labels, slot_probs=None):
    """
    Parse slot labels and words into structured entities.
    Handles BIO tagging (B-entity, I-entity, O).

    Args:
        words: List of words
        slot_labels: List of slot labels (e.g., ['B-weapon_name', 'I-weapon_name', 'O', 'B-target'])
        slot_probs: Optional list of probabilities for each slot prediction

    Returns:
        Tuple of (entities dict, confidences dict)
        - entities: Dictionary mapping entity types to their values
        - confidences: Dictionary mapping entity types to their confidence scores (average of tokens)
    """
    entities = {}
    confidences = {}
    current_entity = None
    current_words = []
    current_probs = []

    for idx, (word, label) in enumerate(zip(words, slot_labels)):
        prob = slot_probs[idx] if slot_probs is not None else None

        if label == 'O':
            # If we were building an entity, save it
            if current_entity:
                entity_key = current_entity.replace('B-', '').replace('I-', '')
                entities[entity_key] = ' '.join(current_words)
                if current_probs:
                    confidences[entity_key] = float(np.mean(current_probs))
                current_entity = None
                current_words = []
                current_probs = []
        elif label.startswith('B-'):
            # Beginning of a new entity
            # Save previous entity if exists
            if current_entity:
                entity_key = current_entity.replace('B-', '').replace('I-', '')
                entities[entity_key] = ' '.join(current_words)
                if current_probs:
                    confidences[entity_key] = float(np.mean(current_probs))

            # Start new entity
            current_entity = label
            current_words = [word]
            current_probs = [prob] if prob is not None else []
        elif label.startswith('I-'):
            # Inside an entity
            if current_entity:
                current_words.append(word)
                if prob is not None:
                    current_probs.append(prob)
            else:
                # This shouldn't happen in well-formed data, but handle it
                current_entity = label
                current_words = [word]
                current_probs = [prob] if prob is not None else []

    # Don't forget the last entity
    if current_entity:
        entity_key = current_entity.replace('B-', '').replace('I-', '')
        entities[entity_key] = ' '.join(current_words)
        if current_probs:
            confidences[entity_key] = float(np.mean(current_probs))

    return entities, confidences


def predict_text(text):
    """Predict intent and slots for input text"""
    global model, tokenizer, args, device, intent_label_lst, slot_label_lst, pad_token_label_id

    if model is None:
        raise Exception("Model not loaded. Call load_model_and_config() first.")

    # Convert input text to dataset
    dataset, words = convert_text_to_dataset(text, tokenizer, args, pad_token_label_id)

    # Create DataLoader
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=1)

    # Predict
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": None,
                "slot_labels_ids": None
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent prediction with confidence
            intent_logits_np = intent_logits.detach().cpu()
            intent_probs = F.softmax(intent_logits_np, dim=1).numpy()
            intent_pred = np.argmax(intent_probs, axis=1)[0]
            intent_confidence = float(intent_probs[0][intent_pred])

            # Slot prediction with confidence
            slot_probs_list = []
            if args.use_crf:
                # For CRF, we get the best path but need to compute probabilities differently
                # CRF decode gives us the best sequence, but we can still get per-token probabilities
                # from the slot_logits before CRF
                slot_preds = np.array(model.crf.decode(slot_logits))
                # Get probabilities from logits for each position
                slot_logits_np = slot_logits.detach().cpu()
                slot_probs_all = F.softmax(slot_logits_np, dim=2).numpy()
                # Extract probability of predicted class for each token
                for j in range(slot_preds.shape[1]):
                    slot_probs_list.append(float(slot_probs_all[0, j, slot_preds[0][j]]))
            else:
                slot_logits_np = slot_logits.detach().cpu()
                slot_probs_all = F.softmax(slot_logits_np, dim=2).numpy()
                slot_preds = np.argmax(slot_probs_all, axis=2)
                # Extract probability of predicted class for each token
                for j in range(slot_preds.shape[1]):
                    slot_probs_list.append(float(slot_probs_all[0, j, slot_preds[0][j]]))

            all_slot_label_mask = batch[3].detach().cpu().numpy()

    # Map slot predictions to labels and collect probabilities
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = []
    slot_confidence_list = []

    for j in range(slot_preds.shape[1]):
        if all_slot_label_mask[0, j] != pad_token_label_id:
            slot_preds_list.append(slot_label_map[slot_preds[0][j]])
            slot_confidence_list.append(slot_probs_list[j])

    # Parse slots into structured entities with confidence scores
    entities, slot_confidences = parse_slots_to_entities(words, slot_preds_list, slot_confidence_list)

    # Calculate overall confidence
    # Overall confidence is the geometric mean of intent confidence and average slot confidence
    if slot_confidence_list:
        avg_slot_confidence = float(np.mean(slot_confidence_list))
        # Use geometric mean to ensure both intent and slots have high confidence
        overall_confidence = float(np.sqrt(intent_confidence * avg_slot_confidence))
    else:
        # If no slots detected, overall confidence is just intent confidence
        overall_confidence = intent_confidence

    # Format output
    result = {
        "intent": intent_label_lst[intent_pred],
        "slots": entities,
        "intent_confidence": intent_confidence,
        "slot_confidences": slot_confidences,
        "overall_confidence": overall_confidence
    }

    return result


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict intent and slots for input text"""
    start_time = datetime.utcnow()
    text = None
    result = None
    error = None

    try:
        # Get input from request
        data = request.get_json()

        if not data or 'text' not in data:
            error = "Missing 'text' field in request body"
            return jsonify({"error": error}), 400

        text = data['text']

        if not text or not text.strip():
            error = "Text cannot be empty"
            return jsonify({"error": error}), 400

        # Make prediction
        result = predict_text(text)

        # Calculate processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log to database
        log_prediction(
            request_text=text,
            response_data=result,
            processing_time_ms=processing_time_ms
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        error = str(e)

        # Calculate processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log error to database
        if text:
            log_prediction(
                request_text=text,
                error=error,
                processing_time_ms=processing_time_ms
            )

        return jsonify({"error": error}), 500


@app.route('/', methods=['GET'])
def index():
    """Welcome endpoint with usage instructions"""
    return jsonify({
        "message": "Chatterlands Intent and Slot Classification API",
        "endpoints": {
            "/health": "GET - Check if the service is healthy",
            "/predict": "POST - Predict intent and slots for input text"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "text": "attack the orc with my iron sword"
            }
        },
        "example_response": {
            "intent": "Attack",
            "slots": {
                "weapon_name": "iron sword",
                "target": "orc"
            },
            "intent_confidence": 0.95,
            "slot_confidences": {
                "weapon_name": 0.92,
                "target": 0.88
            },
            "overall_confidence": 0.91
        }
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./chatterlands_model", type=str, help="Path to saved model")
    parser.add_argument("--port", default=5000, type=int, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host to run the server on")
    parser.add_argument("--db_path", default="predictions.db", type=str, help="Path to SQLite database for logging")
    args_cli = parser.parse_args()

    # Set database path
    db_path = args_cli.db_path

    # Initialize database
    logger.info("Initializing database...")
    init_database()

    # Load model at startup
    logger.info("Loading model...")
    load_model_and_config(model_dir=args_cli.model_dir)

    # Start Flask app
    logger.info(f"Starting server on {args_cli.host}:{args_cli.port}")
    app.run(host=args_cli.host, port=args_cli.port, debug=False)
