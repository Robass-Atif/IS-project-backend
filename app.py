import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import torch
import re
import json
import base64

# Initialize Flask app with CORS for development and production
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://your-frontend-domain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# --- Generate RSA Key Pair ---
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# --- Load Model Safely for Railway ---
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "cybersectony/phishing-email-detection-distilbert_v2.4.1",
        torch_dtype=torch.float32,          # Avoid float16 for CPU-only
        low_cpu_mem_usage=True              # Memory optimization
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Helper Functions ---
def decrypt_data(encrypted_data_b64):
    try:
        encrypted_bytes = base64.b64decode(encrypted_data_b64)
        decrypted_bytes = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return json.loads(decrypted_bytes.decode('utf-8'))
    except Exception as e:
        app.logger.error(f"[Decryption Error]: {str(e)}")
        raise ValueError("Invalid or corrupted encrypted data")

def check_email_address(email):
    suspicious_keywords = ['mail.ru', 'xyz', 'support123', 'loginverify']
    match = re.search(r"@([a-zA-Z0-9.-]+)", email)
    if not match:
        return "⚠️ Invalid email format"
    domain = match.group(1)
    if any(keyword in domain for keyword in suspicious_keywords):
        return f"⚠️ Suspicious domain detected: {domain}"
    elif not domain.endswith(('.com', '.org', '.edu', '.gov')):
        return f"⚠️ Uncommon domain: {domain}"
    else:
        return f"✅ Domain looks fine: {domain}"

def predict_email_body(email_body):
    inputs = tokenizer(email_body, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    probs = predictions[0].tolist()
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }

    max_label = max(labels.items(), key=lambda x: x[1])
    return {
        "prediction": max_label[0],
        "confidence": round(max_label[1], 4),
        "all_probabilities": labels
    }

# --- API Endpoints ---
@app.route('/api/public_key', methods=['GET'])
def get_public_key():
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return jsonify({"public_key": pem.decode('utf-8')}), 200

@app.route('/api/predict', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        if 'encrypted' in data:
            data = decrypt_data(data['encrypted'])

        email = data.get('email')
        body = data.get('body')

        if not email or not body:
            return jsonify({"error": "Both 'email' and 'body' fields are required"}), 400

        email_result = check_email_address(email)
        body_result = predict_email_body(body)

        return jsonify({
            "email_check": email_result,
            "prediction": body_result['prediction'],
            "confidence": body_result['confidence'],
            "all_probabilities": body_result['all_probabilities']
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"[Processing Error]: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# --- Entry Point for Railway or Local ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
