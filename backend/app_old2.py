# app.py (final copy-paste)
import os
import warnings
import json
import uuid
import traceback
import threading
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # load .env

# PDF / embeddings / vector DB imports
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Suppress noisy logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")

# ---------- Gemini init ----------
try:
    import google.generativeai as genai
    import re

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("📝 Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini configured")
except ImportError as e:
    print(f"❌ Missing Gemini package: {e}")
    print("Run: pip install google-generativeai")
    exit(1)
except Exception as e:
    print(f"❌ Failed to configure Gemini: {e}")
    exit(1)

# ---------- Embedding & Chroma init ----------
try:
    # Embedding model (free, small + fast)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer loaded")

    # Chroma client (persistent, new API)
    import chromadb
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    print("✅ ChromaDB client initialized (persist_directory=./chroma_db)")

except ImportError as e:
    print(f"❌ PDF/embedding libs missing: {e}")
    print("Run: pip install PyMuPDF sentence-transformers chromadb")
    exit(1)
except Exception as e:
    print(f"❌ Failed to initialize embeddings/Chroma: {e}")
    exit(1)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

# ---------- Constants ----------
MAX_CHUNKS_PER_UPLOAD = 1500  # cap per upload to avoid huge ingestion
CHUNK_TOKEN_WORDS = 500       # words per chunk (approx)
CHUNK_OVERLAP_WORDS = 50      # overlap words between chunks

# ---------- In-memory/runtime state ----------
active_conversations = {}
chat_history = []
pdf_collections = {}   # map chat_id -> collection object
upload_jobs = {}       # map job_id -> status dict for async uploads

# ---------- Utilities ----------
def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Failed to save {path}: {e}")

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None

# ---------- PDF Processor ----------
class PDFProcessor:
    def __init__(self, embedding_model, chroma_client):
        self.embedding_model = embedding_model
        self.chroma_client = chroma_client

    def extract_text_chunks_from_pdf(self, pdf_path):
        """Extract text per page and chunk into overlapping chunks (words-based)."""
        try:
            doc = fitz.open(pdf_path)
            all_chunks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text") or ""
                text = text.strip()
                if not text:
                    continue

                # normalize whitespace & newlines
                text = " ".join(text.split())

                # split into words and chunk
                words = text.split()
                step = CHUNK_TOKEN_WORDS - CHUNK_OVERLAP_WORDS
                if step <= 0:
                    step = CHUNK_TOKEN_WORDS

                for i in range(0, max(len(words), 1), step):
                    chunk_words = words[i:i + CHUNK_TOKEN_WORDS]
                    if not chunk_words:
                        break
                    chunk_text = " ".join(chunk_words).strip()
                    if chunk_text:
                        all_chunks.append({
                            "content": chunk_text,
                            "page": page_num + 1,
                            "source": Path(pdf_path).name
                        })
            doc.close()
            print(f"📄 Extracted {len(all_chunks)} chunks from {pdf_path}")
            return all_chunks
        except Exception as e:
            print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
            return []

    def create_or_get_chat_collection(self, chat_id):
        """One collection per chat. If not exist, create it."""
        name = f"chat_{chat_id}"
        try:
            # If already in memory, return
            if chat_id in pdf_collections and pdf_collections[chat_id]:
                return pdf_collections[chat_id]

            # If collection exists in Chroma, get it; else create
            existing = None
            try:
                existing = self.chroma_client.get_collection(name)
            except Exception:
                existing = None

            if existing:
                collection = existing
            else:
                collection = self.chroma_client.create_collection(name=name)
            pdf_collections[chat_id] = collection
            return collection
        except Exception as e:
            print(f"❌ Error creating/getting collection {name}: {e}")
            return None

    def add_chunks_to_collection(self, collection, text_chunks):
        """Compute embeddings and add to the provided collection. Returns number of added chunks."""
        try:
            if not text_chunks:
                return 0

            # cap chunks per upload
            if len(text_chunks) > MAX_CHUNKS_PER_UPLOAD:
                print(f"⚠️ Truncating upload to {MAX_CHUNKS_PER_UPLOAD} chunks (was {len(text_chunks)})")
                text_chunks = text_chunks[:MAX_CHUNKS_PER_UPLOAD]

            docs = [c["content"] for c in text_chunks]
            metadatas = [{"page": c["page"], "source": c["source"]} for c in text_chunks]
            ids = [str(uuid.uuid4()) for _ in docs]

            # compute embeddings (sentence-transformers returns numpy array)
            emb = self.embedding_model.encode(docs, show_progress_bar=False)
            emb_list = emb.tolist() if hasattr(emb, "tolist") else [list(e) for e in emb]

            collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=emb_list)
            print(f"✅ Added {len(docs)} chunks to collection {collection.name}")
            return len(docs)
        except Exception as e:
            print(f"❌ Error adding chunks to collection: {e}")
            return 0

    def query_collection_for_context(self, collection, query, n_results=3):
        """Query the collection using embedding nearest neighbors. Returns list of (doc, metadata)."""
        try:
            q_emb = self.embedding_model.encode([query], show_progress_bar=False)
            q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else [list(e) for e in q_emb]
            results = collection.query(query_embeddings=q_emb_list, n_results=n_results)
            # results keys: 'ids', 'documents', 'metadatas', 'distances' (lists inside)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            return list(zip(documents, metadatas))
        except Exception as e:
            print(f"❌ Error querying collection: {e}")
            return []

pdf_processor = PDFProcessor(embedding_model, chroma_client)

# ---------- Chat persistence helpers ----------
def load_chat_history():
    global chat_history
    data = load_json("chat_history.json")
    chat_history[:] = data if data else []
    print(f"📁 Loaded {len(chat_history)} chat sessions")

def save_chat_history():
    save_json("chat_history.json", chat_history)

def load_conversation(chat_id):
    data = load_json(f"memory_{chat_id}.json")
    return data if data else []

def save_conversation(chat_id, conversation):
    save_json(f"memory_{chat_id}.json", conversation)

# ---------- Endpoints ----------
@app.route("/api/chats", methods=["GET"])
def get_chats():
    try:
        return jsonify(chat_history)
    except Exception as e:
        print(f"❌ Error in get_chats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats", methods=["POST"])
def create_chat():
    try:
        chat_id = str(uuid.uuid4())
        new_chat = {
            "id": chat_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
            "has_pdf": False,
            "pdf_count": 0
        }
        chat_history.insert(0, new_chat)
        active_conversations[chat_id] = []
        save_chat_history()
        save_conversation(chat_id, [])
        print(f"✅ Created chat {chat_id}")
        return jsonify(new_chat), 201
    except Exception as e:
        print(f"❌ Error creating chat: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>/upload-pdf", methods=["POST"])
def upload_pdf(chat_id):
    """
    Upload endpoint supports two modes:
      - synchronous (default): process immediately and return when indexing completes
      - async: if query param async=true -> return a job_id immediately with processing: true,
               processing runs in background; poll /api/upload-status/<job_id>
    """
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF file provided (form key: pdf)"}), 400
        pdf_file = request.files["pdf"]
        if pdf_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not pdf_file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File must be a PDF"}), 400

        is_async = request.args.get("async", "false").lower() == "true"

        # Save file to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            pdf_file.save(tmp.name)
            tmp.flush()
        finally:
            tmp.close()

        def process_and_index(temp_path, chat_id, original_name, job_id=None):
            status = {"job_id": job_id, "status": "processing", "started_at": datetime.now().isoformat(), "chunks_added": 0, "error": None}
            if job_id:
                upload_jobs[job_id] = status
            try:
                # extract chunks
                chunks = pdf_processor.extract_text_chunks_from_pdf(temp_path)

                if not chunks:
                    raise Exception("No extractable text found in PDF")

                # create/get collection for chat
                collection = pdf_processor.create_or_get_chat_collection(chat_id)
                if collection is None:
                    raise Exception("Could not create or access vector collection")

                # add chunks (with cap)
                added = pdf_processor.add_chunks_to_collection(collection, chunks)

                # update chat meta
                for chat in chat_history:
                    if chat["id"] == chat_id:
                        chat["has_pdf"] = True
                        chat["pdf_count"] = chat.get("pdf_count", 0) + 1
                        chat["pdf_name"] = original_name
                        chat["updated_at"] = datetime.now().isoformat()
                        break
                save_chat_history()

                status.update({"status": "done", "finished_at": datetime.now().isoformat(), "chunks_added": added})
                if job_id:
                    upload_jobs[job_id] = status
                print(f"✅ PDF {original_name} indexed into chat {chat_id} (chunks_added={added})")
                return {"message": "PDF processed and indexed", "pdf_name": original_name, "chunks_added": added}
            except Exception as e:
                msg = str(e)
                print(f"❌ Error processing PDF: {msg}")
                status.update({"status": "error", "error": msg, "finished_at": datetime.now().isoformat()})
                if job_id:
                    upload_jobs[job_id] = status
                return {"error": msg}
            finally:
                # cleanup temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        if is_async:
            job_id = str(uuid.uuid4())
            thread = threading.Thread(target=process_and_index, args=(tmp.name, chat_id, pdf_file.filename, job_id), daemon=True)
            thread.start()
            upload_jobs[job_id] = {"job_id": job_id, "status": "queued", "started_at": None}
            return jsonify({"message": "Upload accepted", "job_id": job_id, "processing": True}), 202
        else:
            result = process_and_index(tmp.name, chat_id, pdf_file.filename, job_id=None)
            if "error" in result:
                return jsonify({"error": result["error"]}), 500
            return jsonify({"message": result["message"], "pdf_name": result["pdf_name"], "chunks_added": result["chunks_added"], "processing": False})
    except Exception as e:
        print(f"❌ Error in upload_pdf: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload-status/<job_id>", methods=["GET"])
def upload_status(job_id):
    status = upload_jobs.get(job_id)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)

@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    try:
        conversation = load_conversation(chat_id)
        messages = []
        for msg in conversation:
            if msg["role"] == "user":
                messages.append({
                    "id": str(uuid.uuid4()),
                    "text": msg["parts"][0],
                    "sender": "user",
                    "timestamp": datetime.now().isoformat()
                })
            elif msg["role"] == "model":
                messages.append({
                    "id": str(uuid.uuid4()),
                    "text": msg["parts"][0],
                    "sender": "bot",
                    "timestamp": datetime.now().isoformat()
                })
        return jsonify(messages)
    except Exception as e:
        print(f"❌ Error getting chat {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    try:
        global chat_history
        chat_history = [c for c in chat_history if c["id"] != chat_id]
        if chat_id in active_conversations:
            del active_conversations[chat_id]
        # delete collection if exists
        try:
            coll = pdf_collections.get(chat_id)
            if coll:
                chroma_client.delete_collection(coll.name)
                del pdf_collections[chat_id]
        except Exception:
            pass
        try:
            os.remove(f"memory_{chat_id}.json")
        except FileNotFoundError:
            pass
        save_chat_history()
        return jsonify({"message": "Chat deleted successfully"})
    except Exception as e:
        print(f"❌ Error deleting chat {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def send_message(chat_id):
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        if chat_id not in active_conversations:
            active_conversations[chat_id] = load_conversation(chat_id)
        conversation = active_conversations[chat_id]

        # Query PDF collection(s) for context
        system_context_text = ""
        collection = pdf_collections.get(chat_id)
        if collection:
            try:
                # Query for top-K relevant chunks
                results = pdf_processor.query_collection_for_context(collection, user_message, n_results=3)
                if results:
                    system_context_text = "Relevant information from uploaded document(s):\n"
                    for idx, (doc, meta) in enumerate(results, start=1):
                        p = meta.get("page") if isinstance(meta, dict) else meta
                        src = meta.get("source") if isinstance(meta, dict) else None
                        system_context_text += f"[Doc {idx} | source: {src} | page: {p}] {doc}\n\n"
            except Exception as e:
                print(f"⚠️ PDF query failed: {e}")

        # Split multiple questions (preserve original behavior)
        questions = re.split(r"(?<=\?)\s+", user_message)
        all_responses = []

        for question in questions:
            if not question.strip():
                continue

            enhanced_conversation = []

            # If PDF context exists, prepend it as a "user" message
            if system_context_text:
                enhanced_conversation.append({
                    "role": "user",
                    "parts": [
                        f"(Context from uploaded document — use this only if relevant):\n\n{system_context_text}"
                    ]
                })

            # Preserve chat memory
            enhanced_conversation.extend(conversation)

            # Add the new question
            enhanced_conversation.append({"role": "user", "parts": [question.strip()]})

            # Call Gemini
            try:
                response = model.generate_content(enhanced_conversation)
                bot_response = response.text if getattr(response, "text", None) else "I couldn't generate a response."
            except Exception as api_error:
                print(f"❌ Gemini API Error: {api_error}")
                bot_response = f"I encountered an error while processing your request: {str(api_error)}"

            # Save memory (only Q&A, not the big PDF context)
            conversation.append({"role": "user", "parts": [question.strip()]})
            conversation.append({"role": "model", "parts": [bot_response]})
            all_responses.append(bot_response)

            print(f"✅ Generated response (chat {chat_id}): {bot_response[:120]}...")


        # Persist conversation
        save_conversation(chat_id, conversation)

        # Update chat history metadata
        for chat in chat_history:
            if chat["id"] == chat_id:
                chat["updated_at"] = datetime.now().isoformat()
                chat["message_count"] = len(conversation) // 2
                if chat["title"] == "New Chat" and user_message:
                    chat["title"] = user_message[:50] + ("..." if len(user_message) > 50 else "")
                break
        save_chat_history()

        final_response = "\n\n".join(all_responses) if len(all_responses) > 1 else all_responses[0]
        return jsonify({
            "id": str(uuid.uuid4()),
            "text": final_response,
            "sender": "bot",
            "timestamp": datetime.now().isoformat(),
            "used_pdf_context": bool(system_context_text)
        })
    except Exception as e:
        print(f"❌ Error in send_message: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/chats/<chat_id>/rename", methods=["PUT"])
def rename_chat(chat_id):
    try:
        data = request.get_json()
        new_title = data.get("title", "").strip()
        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400
        for chat in chat_history:
            if chat["id"] == chat_id:
                chat["title"] = new_title
                chat["updated_at"] = datetime.now().isoformat()
                break
        save_chat_history()
        return jsonify({"message": "Chat renamed successfully"})
    except Exception as e:
        print(f"❌ Error renaming chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gemini_configured": True,
            "pdf_processing_enabled": True,
            "chats_loaded": len(chat_history),
            "active_pdf_collections": len(pdf_collections),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/chats/<chat_id>/pdfs", methods=["GET"])
def get_chat_pdfs(chat_id):
    """Get PDF information for a specific chat"""
    try:
        # Find the chat
        chat = None
        for c in chat_history:
            if c["id"] == chat_id:
                chat = c
                break
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        if not chat.get("has_pdf", False):
            return jsonify({"pdfs": []})
        
        # Get collection info
        collection = pdf_collections.get(chat_id)
        pdf_info = {
            "name": chat.get("pdf_name", "Unknown"),
            "chunks": 0,
            "status": "active" if collection else "inactive"
        }
        
        if collection:
            try:
                # Get collection count
                result = collection.get()
                pdf_info["chunks"] = len(result["ids"]) if result and "ids" in result else 0
            except Exception as e:
                print(f"Error getting collection info: {e}")
        
        return jsonify({
            "pdfs": [pdf_info],
            "total_pdfs": chat.get("pdf_count", 0),
            "chat_id": chat_id
        })
        
    except Exception as e:
        print(f"❌ Error getting chat PDFs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>/pdfs/<pdf_name>", methods=["DELETE"])
def delete_chat_pdf(chat_id, pdf_name):
    """Delete a specific PDF from a chat (removes collection)"""
    try:
        # Find and update the chat
        for chat in chat_history:
            if chat["id"] == chat_id:
                if chat.get("pdf_name") == pdf_name:
                    # Remove PDF info from chat
                    chat["has_pdf"] = False
                    chat["pdf_count"] = 0
                    if "pdf_name" in chat:
                        del chat["pdf_name"]
                    chat["updated_at"] = datetime.now().isoformat()
                    
                    # Delete collection
                    try:
                        collection = pdf_collections.get(chat_id)
                        if collection:
                            chroma_client.delete_collection(collection.name)
                            del pdf_collections[chat_id]
                            print(f"🗑️  Deleted PDF collection for chat {chat_id}")
                    except Exception as e:
                        print(f"Warning: Could not delete collection: {e}")
                    
                    save_chat_history()
                    return jsonify({"message": "PDF removed successfully"})
                else:
                    return jsonify({"error": "PDF not found in this chat"}), 404
        
        return jsonify({"error": "Chat not found"}), 404
        
    except Exception as e:
        print(f"❌ Error deleting PDF: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- Startup ----------
if __name__ == "__main__":
    print("🚀 Starting Gemini Chat API Server with multi-PDF RAG support...")
    load_chat_history()
    print("📋 Server configuration:")
    print(f"   - Python version: {os.sys.version}")
    print(f"   - Flask CORS enabled: ✅")
    print(f"   - Gemini API key configured: ✅")
    print(f"   - PDF processing enabled: ✅")
    print(f"   - Vector DB: ChromaDB (persist_directory=./chroma_db) ✅")
    print(f"   - Embedding model: all-MiniLM-L6-v2 ✅")
    print(f"   - Chat history loaded: {len(chat_history)} chats")
    print("🌐 Starting server on http://localhost:5000")
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
