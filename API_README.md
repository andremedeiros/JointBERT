# Chatterlands Intent and Slot Classification API

A Flask-based web service for running the Chatterlands JointBERT classifier. This service accepts text phrases and returns structured intent and slot predictions.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

Start the server with:

```bash
python app.py
```

By default, the service runs on `http://0.0.0.0:5000`

### Command-line Options

- `--model_dir`: Path to the trained model (default: `./chatterlands_model`)
- `--port`: Port to run the server on (default: `5000`)
- `--host`: Host to bind to (default: `0.0.0.0`)

Example:
```bash
python app.py --model_dir ./chatterlands_model --port 8080 --host localhost
```

## API Endpoints

### GET /
Welcome endpoint with API documentation

**Response:**
```json
{
  "message": "Chatterlands Intent and Slot Classification API",
  "endpoints": {
    "/health": "GET - Check if the service is healthy",
    "/predict": "POST - Predict intent and slots for input text"
  }
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
Predict intent and slots for input text

**Request Body:**
```json
{
  "text": "attack the orc with my iron sword"
}
```

**Response:**
```json
{
  "intent": "Attack",
  "slots": {
    "weapon_name": "iron sword",
    "target": "orc"
  }
}
```

**Error Response:**
```json
{
  "error": "Missing 'text' field in request body"
}
```

## Usage Examples

### Using cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "attack the orc with my iron sword"}'
```

### Using Python requests

```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "attack the orc with my iron sword"}

response = requests.post(url, json=data)
result = response.json()

print(f"Intent: {result['intent']}")
print(f"Slots: {result['slots']}")
```

### Using JavaScript fetch

```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'attack the orc with my iron sword'
  })
});

const result = await response.json();
console.log('Intent:', result.intent);
console.log('Slots:', result.slots);
```

## Response Format

The API returns structured JSON with two main fields:

- **intent**: The predicted intent (e.g., "Attack", "Move", "Talk")
- **slots**: A dictionary mapping slot types to their extracted values

The slots dictionary uses BIO (Begin-Inside-Outside) tagging internally to handle multi-word entities correctly. For example:
- "iron sword" is recognized as a single `weapon_name` entity
- Multiple entities of the same type can be distinguished

## Notes

- The model is loaded once at startup for efficiency
- The service uses the device priority: MPS (Mac) > CUDA (GPU) > CPU
- Empty or missing text will return a 400 error
- All errors return appropriate HTTP status codes and error messages
