# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PyPDF2 import PdfReader

# app = Flask(__name__)
# CORS(app)  # allows frontend to connect

# # Store document text globally for now
# document_text = ""

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global document_text
#     file = request.files.get('document')
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     reader = PdfReader(file)
#     document_text = "\n".join(page.extract_text() or "" for page in reader.pages)
#     return jsonify({"message": "File uploaded successfully!"})

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global document_text
#     if not document_text:
#         return jsonify({"answer": "Please upload a document first.", "sources": []})

#     query = request.json.get("query", "").lower()

#     # Fake search logic — replace with LLM or embedding-based search later
#     if "refund" in query:
#         return jsonify({
#             "answer": "Refunds are processed within 30 days.",
#             "sources": [{"text": "Section 3: Refund Policy"}]
#         })
#     elif "leave" in query:
#         return jsonify({
#             "answer": "Employees get 20 paid leaves annually.",
#             "sources": [{"text": "Section 2: Leave Policy"}]
#         })
#     else:
#         return jsonify({
#             "answer": "No relevant information found.",
#             "sources": []
#         })

# if __name__ == '__main__':
#     app.run(port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import tempfile

from PyPDF2 import PdfReader
# You can extend with: from docx import Document

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# For LLM: from openai import OpenAI  (if using OpenAI LLM/embedding)

app = Flask(__name__)
CORS(app)

# Load embedding model (choose small for demo, or a larger one)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# Our "corpus": [(chunk_text, metadata_dict)]
doc_chunks = []
embeddings = None  # NumPy array of chunk vectors
clause_metadata = []  # For mapping chunks to their origins

def chunk_text(text, max_length=512):
    """Chunk text by paragraphs and limit per chunk."""
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    chunk = ""
    for para in paras:
        if len(chunk) + len(para) < max_length:
            chunk += ("\n" if chunk else "") + para
        else:
            if chunk: chunks.append(chunk.strip())
            chunk = para
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def embed_chunks(text_chunks):
    """Embed list of text chunks using ST model. Returns np.array [n_chunks x d]"""
    return np.array(embedding_model.encode(text_chunks, show_progress_bar=False, normalize_embeddings=True))

def build_faiss_index(emb_matrix):
    """Builds FAISS index for L2 search (vectors must be normalized for cosine similarity)"""
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product = Cosine (with normalized vectors)
    index.add(emb_matrix)
    return index

@app.route('/upload', methods=['POST'])
def upload_file():
    global doc_chunks, embeddings, clause_metadata

    uploaded_file = request.files.get('document')
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Only handling PDF for now; add other formats as needed.
    suffix = os.path.splitext(uploaded_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tempf:
        uploaded_file.save(tempf.name)
        if suffix == ".pdf":
            reader = PdfReader(tempf.name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # add for .docx, .eml, etc.
        else:
            return jsonify({"error": "Unsupported filetype"}), 400

    # Chunk document, keep section heading if heuristic possible.
    chunks = chunk_text(text)

    # Optionally add section/para numbers as metadata
    doc_chunks = []
    clause_metadata = []
    for idx, ch in enumerate(chunks):
        doc_chunks.append(ch)
        clause_metadata.append({"clause_id": idx, "section": f"Chunk {idx+1}"})

    # Embed all
    embeddings = embed_chunks(doc_chunks)
    # Build/refresh vector index
    global faiss_index
    faiss_index = build_faiss_index(embeddings)

    return jsonify({"message": f"File uploaded and indexed! {len(doc_chunks)} chunks."})

def find_relevant_chunks(query, top_k=5):
    """Embed query and semantically search indexed doc."""
    query_vec = embedding_model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(doc_chunks):
            results.append({
                "score": float(dist),
                "text": doc_chunks[idx],
                "meta": clause_metadata[idx]
            })
    return results

def mock_llm_decision(query, top_clauses):
    """Simulate LLM's reasoning and output. Replace with real LLM call in production."""
    # In real system: Compose prompt with structured entity extraction, retrieved clauses, and ask LLM for answer.
    # Here, imitate with rules for demo.
    ql = query.lower()
    # Simulate extracting features
    decision, amount, justification = "rejected", None, "No coverage found for your query."
    for clause in top_clauses:
        txt = clause["text"].lower()
        if "knee surgery" in txt or "surgery" in ql:
            decision = "approved"
            amount = "100,000"
            justification = f"Knee surgery is included under {clause['meta']['section']}"
            break
        elif "leave" in ql and "leave" in txt:
            decision = "approved"
            amount = None
            justification = f"Leave policy found under {clause['meta']['section']}"
            break
        elif "refund" in ql and "refund" in txt:
            decision = "approved"
            amount = None
            justification = f"Refund is sanctioned as per {clause['meta']['section']}"
            break

    # Return all clause refs for explainability
    clause_refs = [{"section": c['meta']['section'], "text": c['text']} for c in top_clauses]
    return {
        "decision": decision,
        "amount": amount,
        "justification": justification,
        "clause_refs": clause_refs
    }

@app.route('/ask', methods=['POST'])
def ask_question():
    if not embeddings is not None or len(doc_chunks) == 0:
        return jsonify({
            "answer": "Please upload a document first.",
            "sources": []
        })

    query = request.json.get("query", "")
    # 1. Find semantically relevant chunks
    top_chunks = find_relevant_chunks(query, top_k=5)

    if not top_chunks:
        return jsonify({
            "decision": "rejected",
            "amount": None,
            "justification": "No relevant clauses found in document.",
            "clause_refs": []
        })

    # 2. LLM-based reasoning (simulated here)
    result = mock_llm_decision(query, top_chunks)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
