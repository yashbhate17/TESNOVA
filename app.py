from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)  # allows frontend to connect

# Store document text globally for now
document_text = ""

@app.route('/upload', methods=['POST'])
def upload_file():
    global document_text
    file = request.files.get('document')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    reader = PdfReader(file)
    document_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return jsonify({"message": "File uploaded successfully!"})

@app.route('/ask', methods=['POST'])
def ask_question():
    global document_text
    if not document_text:
        return jsonify({"answer": "Please upload a document first.", "sources": []})

    query = request.json.get("query", "").lower()

    # Fake search logic — replace with LLM or embedding-based search later
    if "refund" in query:
        return jsonify({
            "answer": "Refunds are processed within 30 days.",
            "sources": [{"text": "Section 3: Refund Policy"}]
        })
    elif "leave" in query:
        return jsonify({
            "answer": "Employees get 20 paid leaves annually.",
            "sources": [{"text": "Section 2: Leave Policy"}]
        })
    else:
        return jsonify({
            "answer": "No relevant information found.",
            "sources": []
        })

if __name__ == '__main__':
    app.run(port=5000)