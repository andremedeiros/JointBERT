#!/usr/bin/env python3
"""
Backfill script to recalculate confidence scores for existing predictions in the database.

This script loads the model and reprocesses all predictions that don't have confidence scores,
updating the database with the calculated intent_confidence and overall_confidence values.

Usage:
    python backfill_confidence_scores.py --db_path predictions.db --model_dir ./chatterlands_model
    python backfill_confidence_scores.py --db_path predictions.db --model_dir ./chatterlands_model --batch_size 32
    python backfill_confidence_scores.py --db_path predictions.db --model_dir ./chatterlands_model --force  # Recalculate all
"""

import os
import logging
import sqlite3
import argparse
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn.functional as F

from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES

logger = logging.getLogger(__name__)


class PredConfig:
    """Configuration for prediction"""
    def __init__(self, model_dir="./chatterlands_model", batch_size=1, no_cuda=False):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.no_cuda = no_cuda


def get_device(pred_config):
    """Get the best available device"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built() and not pred_config.no_cuda:
        return "mps"
    elif torch.cuda.is_available() and not pred_config.no_cuda:
        return "cuda"
    else:
        return "cpu"


def load_model_and_config(model_dir):
    """Load model and configuration"""
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

    return model, tokenizer, args, device, intent_label_lst, slot_label_lst, pad_token_label_id


def convert_text_to_dataset(text, tokenizer, args, pad_token_label_id):
    """Convert input text to TensorDataset"""
    words = text.strip().split()

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
            word_tokens = [unk_token]
        tokens.extend(word_tokens)
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


def calculate_confidence_scores(text, model, tokenizer, args, device, intent_label_lst, slot_label_lst, pad_token_label_id):
    """Calculate confidence scores for a given text"""
    try:
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
                    slot_preds = np.array(model.crf.decode(slot_logits))
                    slot_logits_np = slot_logits.detach().cpu()
                    slot_probs_all = F.softmax(slot_logits_np, dim=2).numpy()
                    for j in range(slot_preds.shape[1]):
                        slot_probs_list.append(float(slot_probs_all[0, j, slot_preds[0][j]]))
                else:
                    slot_logits_np = slot_logits.detach().cpu()
                    slot_probs_all = F.softmax(slot_logits_np, dim=2).numpy()
                    slot_preds = np.argmax(slot_probs_all, axis=2)
                    for j in range(slot_preds.shape[1]):
                        slot_probs_list.append(float(slot_probs_all[0, j, slot_preds[0][j]]))

                all_slot_label_mask = batch[3].detach().cpu().numpy()

        # Collect slot confidences for non-padding tokens
        slot_confidence_list = []
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[0, j] != pad_token_label_id:
                slot_confidence_list.append(slot_probs_list[j])

        # Calculate overall confidence
        if slot_confidence_list:
            avg_slot_confidence = float(np.mean(slot_confidence_list))
            overall_confidence = float(np.sqrt(intent_confidence * avg_slot_confidence))
        else:
            overall_confidence = intent_confidence

        return intent_confidence, overall_confidence

    except Exception as e:
        logger.error(f"Error calculating confidence for text '{text}': {str(e)}")
        return None, None


def backfill_database(db_path, model_dir, force=False, batch_commit=100):
    """Backfill confidence scores for existing predictions"""
    logger.info("Loading model and configuration...")
    model, tokenizer, args, device, intent_label_lst, slot_label_lst, pad_token_label_id = load_model_and_config(model_dir)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get predictions that need confidence scores
    if force:
        logger.info("Force mode: Recalculating confidence scores for all predictions")
        cursor.execute("SELECT id, request_text FROM predictions WHERE error IS NULL ORDER BY id")
    else:
        logger.info("Fetching predictions without confidence scores...")
        cursor.execute("""
            SELECT id, request_text FROM predictions
            WHERE error IS NULL AND (intent_confidence IS NULL OR overall_confidence IS NULL)
            ORDER BY id
        """)

    predictions = cursor.fetchall()
    total_predictions = len(predictions)

    if total_predictions == 0:
        logger.info("No predictions need backfilling!")
        conn.close()
        return

    logger.info(f"Found {total_predictions} predictions to process")

    # Process each prediction
    updated_count = 0
    failed_count = 0

    for idx, (pred_id, request_text) in enumerate(tqdm(predictions, desc="Backfilling")):
        intent_confidence, overall_confidence = calculate_confidence_scores(
            request_text, model, tokenizer, args, device,
            intent_label_lst, slot_label_lst, pad_token_label_id
        )

        if intent_confidence is not None and overall_confidence is not None:
            cursor.execute("""
                UPDATE predictions
                SET intent_confidence = ?, overall_confidence = ?
                WHERE id = ?
            """, (intent_confidence, overall_confidence, pred_id))
            updated_count += 1
        else:
            failed_count += 1

        # Commit in batches
        if (idx + 1) % batch_commit == 0:
            conn.commit()
            logger.info(f"Committed batch: {idx + 1}/{total_predictions}")

    # Final commit
    conn.commit()
    conn.close()

    logger.info("=" * 60)
    logger.info(f"Backfill complete!")
    logger.info(f"Total processed: {total_predictions}")
    logger.info(f"Successfully updated: {updated_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("=" * 60)


if __name__ == '__main__':
    init_logger()

    parser = argparse.ArgumentParser(description="Backfill confidence scores for existing predictions")
    parser.add_argument("--db_path", default="predictions.db", type=str, help="Path to SQLite database")
    parser.add_argument("--model_dir", default="./chatterlands_model", type=str, help="Path to saved model")
    parser.add_argument("--force", action="store_true", help="Recalculate confidence for all predictions (not just missing)")
    parser.add_argument("--batch_commit", default=100, type=int, help="Number of records to process before committing")

    args = parser.parse_args()

    backfill_database(args.db_path, args.model_dir, args.force, args.batch_commit)
