import re
from typing import List, Dict, Any

import os
import warnings
import json
import uuid
import traceback
import threading
import tempfile
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import tiktoken

load_dotenv()  # load .env

# PyMuPDF for fast text extraction
import fitz  # PyMuPDF
import pdfplumber

from sentence_transformers import SentenceTransformer
import chromadb

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
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer loaded")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print("✅ ChromaDB client initialized (persist_directory=./chroma_db)")

except ImportError as e:
    print(f"❌ PDF/embedding libs missing: {e}")
    print("Run: pip install PyMuPDF sentence-transformers chromadb pdfplumber")
    exit(1)
except Exception as e:
    print(f"❌ Failed to initialize embeddings/Chroma: {e}")
    exit(1)

app = Flask(__name__)
CORS(app)

MAX_CHUNKS_PER_UPLOAD = 1500

active_conversations = {}
chat_history = []
pdf_collections = {}
upload_jobs = {}

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

class PDFProcessor:
    def __init__(self, embedding_model, chroma_client):
        self.embedding_model = embedding_model
        self.chroma_client = chroma_client
        
        # Try to import docling for fallback (optional)
        self.docling_available = False
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling_core.types.doc import TextItem, TableItem
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(do_ocr=True)
                    )
                }
            )
            self.docling_available = True
            print("✅ Docling available as fallback")
        except ImportError:
            print("⚠️ Docling not available - using PyMuPDF only")
        
        # Tokenizer for estimating tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_string(self, string: str) -> int:
        try:
            return len(self.tokenizer.encode(string))
        except Exception:
            return len(string.split())

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text.strip()

    def create_overlapping_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """Create overlapping chunks to avoid losing context at boundaries"""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(words) - overlap:
                break
                
        return chunks

    def extract_text_pymupdf(self, pdf_path: str) -> Dict[int, str]:
        """Fast text extraction using PyMuPDF"""
        page_texts = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text and text.strip():
                    cleaned_text = self.clean_text(text)
                    page_texts[page_num + 1] = cleaned_text  # 1-indexed pages
            doc.close()
            print(f"🚀 PyMuPDF extracted text from {len(page_texts)} pages")
        except Exception as e:
            print(f"❌ PyMuPDF extraction error: {e}")
        
        return page_texts

    def extract_text_docling_fallback(self, pdf_path: str) -> Dict[int, str]:
        """Fallback docling extraction for complex PDFs"""
        page_texts = {}
        if not self.docling_available:
            return page_texts
            
        try:
            from docling_core.types.doc import TextItem
            conv_result = self.converter.convert(pdf_path)
            
            for item in conv_result.document.iterate_items():
                if isinstance(item, TextItem):
                    text = getattr(item, "text", "")
                    page = getattr(item, "page_number", 1)
                    
                    if text and text.strip():
                        cleaned_text = self.clean_text(text)
                        if page not in page_texts:
                            page_texts[page] = ""
                        page_texts[page] += " " + cleaned_text
            
            print(f"🔄 Docling fallback extracted text from {len(page_texts)} pages")
        except Exception as e:
            print(f"❌ Docling fallback error: {e}")
        
        return page_texts

    def should_use_docling_fallback(self, page_texts: Dict[int, str], threshold: int = 50) -> bool:
        """Determine if PyMuPDF extraction was insufficient"""
        if not page_texts:
            return True
        
        poor_pages = sum(1 for text in page_texts.values() if len(text.strip()) < threshold)
        total_pages = len(page_texts)
        
        # Use fallback if more than 30% of pages have very little text
        return poor_pages > (total_pages * 0.3)

    def extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract tables using pdfplumber (keeping existing logic)"""
        table_chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page_obj in enumerate(pdf.pages, start=1):
                    tables = page_obj.extract_tables() or []
                    
                    for table_idx, table in enumerate(tables):
                        if len(table) < 2:
                            continue
                            
                        # Extract table with better formatting
                        headers = table[0] if table[0] else []
                        table_content = []
                        
                        # Create table summary
                        table_summary = f"Table on page {page_num}:\n"
                        if headers:
                            table_summary += "Headers: " + " | ".join([str(h) for h in headers if h]) + "\n"
                        
                        # Process table rows
                        for row in table[1:]:
                            if not any(row):
                                continue
                            row_data = []
                            for j, cell in enumerate(row):
                                if cell and j < len(headers) and headers[j]:
                                    row_data.append(f"{headers[j]}: {cell}")
                            if row_data:
                                table_content.append(" | ".join(row_data))
                        
                        if table_content:
                            full_table = table_summary + "\n".join(table_content)
                            table_chunks.append({
                                "content": self.clean_text(full_table),
                                "page": page_num,
                                "source": pdf_path,
                                "type": "table",
                                "chunk_id": f"table_page_{page_num}_{table_idx}"
                            })
        except Exception as e:
            print(f"⚠️ Table extraction error: {e}")
        
        return table_chunks

    def extract_text_chunks_from_pdf(self, pdf_path: str, chunk_size=1500):
        """Enhanced PDF processing with PyMuPDF first, docling fallback - NO MORE 'multiple' pages"""
        chunks = []
        
        print(f"🔍 Processing PDF: {pdf_path}")
        start_time = time.time()
        
        # --- Step 1: Fast text extraction with PyMuPDF ---
        page_texts = self.extract_text_pymupdf(pdf_path)
        
        # --- Step 2: Check if we need docling fallback ---
        if self.should_use_docling_fallback(page_texts) and self.docling_available:
            print("⚠️ PyMuPDF extraction insufficient, trying docling fallback...")
            fallback_texts = self.extract_text_docling_fallback(pdf_path)
            if fallback_texts:
                page_texts = fallback_texts
        
        # --- Step 3: Process extracted text into page-level chunks ONLY ---
        for page_num, page_content in page_texts.items():
            # Create overlapping chunks for this specific page
            page_chunks = self.create_overlapping_chunks(page_content, chunk_size, overlap=200)
            
            for i, chunk in enumerate(page_chunks):
                if len(chunk.strip()) > 50:  # Only keep substantial chunks
                    chunks.append({
                        "content": chunk,
                        "page": page_num,  # EXACT page number
                        "source": pdf_path,
                        "type": "text",
                        "chunk_id": f"page_{page_num}_chunk_{i}"
                    })
        
        # --- REMOVED: Document-level chunks with "multiple" pages ---
        # We no longer create cross-page chunks to avoid ambiguous sources
        
        # --- Step 4: Extract tables with pdfplumber ---
        table_chunks = self.extract_tables_pdfplumber(pdf_path)
        chunks.extend(table_chunks)
        
        extraction_time = time.time() - start_time
        print(f"✅ Created {len(chunks)} total chunks in {extraction_time:.2f}s")
        
        return chunks

    def create_or_get_chat_collection(self, chat_id):
        """Enhanced collection management"""
        name = f"chat_{chat_id}"
        try:
            if chat_id in pdf_collections:
                return pdf_collections[chat_id]
            
            try:
                collection = self.chroma_client.get_collection(name)
                print(f"📂 Retrieved existing collection: {name}")
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=name,
                    metadata={"description": f"PDF collection for chat {chat_id}"}
                )
                print(f"📂 Created new collection: {name}")
                
            pdf_collections[chat_id] = collection
            return collection
        except Exception as e:
            print(f"❌ Error with collection {name}: {e}")
            return None

    def add_chunks_to_collection(self, collection, text_chunks):
        """Enhanced chunk storage with better metadata"""
        if not text_chunks:
            return 0
        
        # Limit chunks to avoid overwhelming the system
        if len(text_chunks) > MAX_CHUNKS_PER_UPLOAD:
            print(f"⚠️ Limiting to {MAX_CHUNKS_PER_UPLOAD} chunks (was {len(text_chunks)})")
            text_chunks = text_chunks[:MAX_CHUNKS_PER_UPLOAD]
        
        docs = [str(c["content"]) for c in text_chunks]
        metadatas = [
            {
                "page": c.get("page", 1),
                "source": Path(c.get("source", "unknown")).name,
                "type": c.get("type", "text"),
                "chunk_id": c.get("chunk_id", f"chunk_{i}"),
                "word_count": len(c["content"].split()),
                "char_count": len(c["content"])
            }
            for i, c in enumerate(text_chunks)
        ]
        ids = [f"{c.get('chunk_id', f'chunk_{i}')}_{uuid.uuid4().hex[:8]}" for i, c in enumerate(text_chunks)]
        
        try:
            emb = self.embedding_model.encode(docs, show_progress_bar=False)
            emb_list = emb.tolist() if hasattr(emb, "tolist") else [list(e) for e in emb]
            
            collection.add(
                documents=docs, 
                metadatas=metadatas, 
                ids=ids, 
                embeddings=emb_list
            )
            
            print(f"✅ Added {len(docs)} chunks to collection {collection.name}")
            return len(docs)
            
        except Exception as e:
            print(f"❌ Error adding chunks to collection: {e}")
            return 0

    def enhanced_query_collection(self, collection, query: str, n_results: int = 8) -> List[tuple]:
        """Enhanced querying with multiple search strategies"""
        all_results = []
        
        try:
            # Strategy 1: Direct semantic search
            q_emb = self.embedding_model.encode([query], show_progress_bar=False)
            q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else [list(e) for e in q_emb]
            
            results = collection.query(
                query_embeddings=q_emb_list, 
                n_results=n_results
            )
            
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                all_results.append((doc, meta, dist))
            
            # Strategy 2: Keyword-based search for exact matches
            query_words = query.lower().split()
            keyword_results = collection.get()
            
            if keyword_results and "documents" in keyword_results:
                for i, doc in enumerate(keyword_results["documents"]):
                    if any(word in doc.lower() for word in query_words if len(word) > 3):
                        meta = keyword_results["metadatas"][i] if i < len(keyword_results["metadatas"]) else {}
                        # Add with high relevance score for direct matches
                        all_results.append((doc, meta, 0.05))
            
            # Remove duplicates and sort by relevance
            seen = set()
            unique_results = []
            for doc, meta, dist in all_results:
                doc_signature = doc[:100]  # Use first 100 chars as signature
                if doc_signature not in seen:
                    seen.add(doc_signature)
                    unique_results.append((doc, meta))
            
            # Sort by distance (lower is better) and return top results
            unique_results.sort(key=lambda x: all_results[[r[0] for r in all_results].index(x[0])][2])
            return unique_results[:n_results]
            
        except Exception as e:
            print(f"❌ Error querying collection: {e}")
            return []

    def query_collection_for_context(self, collection, query, n_results=10):
        """Use enhanced querying with more results for simple queries"""
        return self.enhanced_query_collection(collection, query, n_results)

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
            "pdf_count": 0,
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
    """Upload endpoint with faster PyMuPDF processing"""
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF file provided (form key: pdf)"}), 400
        pdf_file = request.files["pdf"]
        if pdf_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not pdf_file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File must be a PDF"}), 400

        is_async = request.args.get("async", "false").lower() == "true"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            pdf_file.save(tmp.name)
            tmp.flush()
        finally:
            tmp.close()

        def process_and_index(temp_path, chat_id, original_name, job_id=None):
            status = {
                "job_id": job_id,
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "chunks_added": 0,
                "error": None,
            }
            if job_id:
                upload_jobs[job_id] = status
            try:
                chunks = pdf_processor.extract_text_chunks_from_pdf(temp_path)

                if not chunks:
                    raise Exception("No extractable text found in PDF")

                collection = pdf_processor.create_or_get_chat_collection(chat_id)
                if collection is None:
                    raise Exception("Could not create or access vector collection")

                added = pdf_processor.add_chunks_to_collection(collection, chunks)

                for chat in chat_history:
                    if chat["id"] == chat_id:
                        chat["has_pdf"] = True
                        chat["pdf_count"] = chat.get("pdf_count", 0) + 1
                        chat["pdf_name"] = original_name
                        chat["updated_at"] = datetime.now().isoformat()
                        break
                save_chat_history()

                status.update(
                    {
                        "status": "done",
                        "finished_at": datetime.now().isoformat(),
                        "chunks_added": added,
                    }
                )
                if job_id:
                    upload_jobs[job_id] = status
                print(
                    f"✅ PDF {original_name} indexed into chat {chat_id} (chunks_added={added})"
                )
                return {
                    "message": "PDF processed and indexed",
                    "pdf_name": original_name,
                    "chunks_added": added,
                }
            except Exception as e:
                msg = str(e)
                print(f"❌ Error processing PDF: {msg}")
                status.update(
                    {
                        "status": "error",
                        "error": msg,
                        "finished_at": datetime.now().isoformat(),
                    }
                )
                if job_id:
                    upload_jobs[job_id] = status
                return {"error": msg}
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        if is_async:
            job_id = str(uuid.uuid4())
            thread = threading.Thread(
                target=process_and_index,
                args=(tmp.name, chat_id, pdf_file.filename, job_id),
                daemon=True,
            )
            thread.start()
            upload_jobs[job_id] = {"job_id": job_id, "status": "queued", "started_at": None}
            return jsonify(
                {"message": "Upload accepted", "job_id": job_id, "processing": True}
            ), 202
        else:
            result = process_and_index(tmp.name, chat_id, pdf_file.filename, job_id=None)
            if "error" in result:
                return jsonify({"error": result["error"]}), 500
            return jsonify(
                {
                    "message": result["message"],
                    "pdf_name": result["pdf_name"],
                    "chunks_added": result["chunks_added"],
                    "processing": False,
                }
            )
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
                messages.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": msg["parts"][0],
                        "sender": "user",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            elif msg["role"] == "model":
                messages.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": msg["parts"][0],
                        "sender": "bot",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
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

        # Improved query handling for simple vs complex queries
        is_simple_query = len(user_message.split()) <= 10 and not any(
            word in user_message.lower() for word in ['analyze', 'compare', 'extract', 'summarize', 'detailed']
        )

        system_context_text = ""
        source_references = []
        collection = pdf_collections.get(chat_id)
        
        if collection:
            try:
                # Adjust search strategy based on query complexity
                n_results = 15 if is_simple_query else 10
                results = pdf_processor.query_collection_for_context(
                    collection, user_message, n_results=n_results
                )
                
                if results:
                    system_context_text = "Here is relevant information from the document:\n\n"
                    
                    for idx, (doc, meta) in enumerate(results, start=1):
                        # FIXED: Handle exact pages properly, no more "multiple"
                        if isinstance(meta, dict):
                            page = meta.get("page", "unknown")
                            source = meta.get("source", "Unknown")
                            doc_type = meta.get("type", "text")
                        else:
                            page = "unknown"
                            source = "Unknown" 
                            doc_type = "text"
                        
                        # Build source references with exact page info
                        source_references.append({
                            "id": idx,
                            "content": str(doc)[:800],
                            "full_content": str(doc),
                            "page": page,  # This will be exact page number, not "multiple"
                            "source": source,
                            "type": doc_type,
                        })
                        
                        # Build context without explicit reference numbers
                        system_context_text += f"[Page {page}, {doc_type}, Source: {source}]\n{doc}\n\n"
                        
                    print(f"🔍 Found {len(results)} relevant chunks for {'simple' if is_simple_query else 'complex'} query: {user_message[:50]}...")
                    
            except Exception as e:
                print(f"⚠️ PDF query failed: {e}")

        questions = re.split(r"(?<=\?)\s+", user_message)
        all_responses = []

        for question in questions:
            if not question.strip():
                continue

            enhanced_conversation = []

            if system_context_text:
                # FIXED: Don't ask model to cite reference numbers, just use natural context
                if is_simple_query:
                    prompt = f"""You are a helpful assistant. Answer this question directly and concisely using the information provided.

{system_context_text}

Question: {question.strip()}

Instructions:
- Give a direct answer based on the information above
- If the answer isn't in the provided information, say "I don't see that information in the document"
- Don't mention reference numbers or sources - just answer naturally"""
                else:
                    prompt = f"""You are a helpful assistant. Answer this question comprehensively using the information provided.

{system_context_text}

Question: {question.strip()}

Instructions: 
- Use the information above to provide a detailed answer
- If some information is missing, mention what you couldn't find
- Answer naturally without citing specific reference numbers"""

                enhanced_conversation.append({
                    "role": "user",
                    "parts": [prompt]
                })

            # Add recent conversation context (less for simple queries)
            context_length = 4 if is_simple_query else 6
            recent_conversation = conversation[-context_length:] if len(conversation) > context_length else conversation
            enhanced_conversation.extend(recent_conversation)
            enhanced_conversation.append({"role": "user", "parts": [question.strip()]})

            try:
                response = model.generate_content(enhanced_conversation)
                bot_response = response.text if getattr(response, "text", None) else "I couldn't generate a response."
            except Exception as api_error:
                print(f"❌ Gemini API Error: {api_error}")
                bot_response = f"I encountered an error while processing your request: {str(api_error)}"

            conversation.append({"role": "user", "parts": [question.strip()]})
            conversation.append({"role": "model", "parts": [bot_response]})
            all_responses.append(bot_response)

        save_conversation(chat_id, conversation)

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
            "used_pdf_context": bool(system_context_text),
            "source_references": source_references,  # Now contains exact pages
            "query_type": "simple" if is_simple_query else "complex"
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
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "gemini_configured": True,
                "pdf_processing_enabled": True,
                "pdf_engine": "PyMuPDF + pdfplumber (docling fallback)" if pdf_processor.docling_available else "PyMuPDF + pdfplumber",
                "chats_loaded": len(chat_history),
                "active_pdf_collections": len(pdf_collections),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/chats/<chat_id>/pdfs", methods=["GET"])
def get_chat_pdfs(chat_id):
    """Get PDF information for a specific chat"""
    try:
        chat = None
        for c in chat_history:
            if c["id"] == chat_id:
                chat = c
                break

        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        if not chat.get("has_pdf", False):
            return jsonify({"pdfs": [], "total_pdfs": 0, "chat_id": chat_id})

        collection = pdf_collections.get(chat_id)
        chunks_count = 0
        status = "inactive"

        if collection:
            try:
                result = collection.get()
                if result and "ids" in result:
                    chunks_count = len(result["ids"])
                    status = "active"
            except Exception as e:
                print(f"Warning: Could not get collection info: {e}")
                status = "error"

        pdf_info = {
            "name": chat.get("pdf_name", "Unknown Document"),
            "chunks": chunks_count,
            "status": status,
        }

        return jsonify(
            {
                "pdfs": [pdf_info],
                "total_pdfs": chat.get("pdf_count", 0),
                "chat_id": chat_id,
            }
        )

    except Exception as e:
        print(f"❌ Error getting chat PDFs for {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/<chat_id>/pdfs/<pdf_name>", methods=["DELETE"])
def delete_chat_pdf(chat_id, pdf_name):
    """Delete a specific PDF from a chat"""
    try:
        chat_found = False
        for chat in chat_history:
            if chat["id"] == chat_id:
                chat_found = True
                current_pdf_name = chat.get("pdf_name", "")

                if current_pdf_name == pdf_name or pdf_name in current_pdf_name:
                    chat["has_pdf"] = False
                    chat["pdf_count"] = 0
                    if "pdf_name" in chat:
                        del chat["pdf_name"]
                    chat["updated_at"] = datetime.now().isoformat()

                    print(f"🗑️  Removing PDF '{pdf_name}' from chat {chat_id}")

                    try:
                        collection = pdf_collections.get(chat_id)
                        if collection:
                            chroma_client.delete_collection(collection.name)
                            print(f"✅ Deleted ChromaDB collection: {collection.name}")

                        if chat_id in pdf_collections:
                            del pdf_collections[chat_id]
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete collection: {e}")

                    save_chat_history()

                    return jsonify(
                        {
                            "message": "PDF removed successfully",
                            "chat_id": chat_id,
                            "pdf_name": pdf_name,
                        }
                    )
                else:
                    return (
                        jsonify({"error": f"PDF '{pdf_name}' not found in this chat"}),
                        404,
                    )

        if not chat_found:
            return jsonify({"error": "Chat not found"}), 404

    except Exception as e:
        print(f"❌ Error deleting PDF '{pdf_name}' from chat {chat_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/pdfs", methods=["GET"])
def get_all_pdfs():
    """Get all PDFs across all chats"""
    try:
        all_pdfs = []
        for chat in chat_history:
            if chat.get("has_pdf", False):
                collection = pdf_collections.get(chat["id"])
                chunks_count = 0
                status = "inactive"

                if collection:
                    try:
                        result = collection.get()
                        if result and "ids" in result:
                            chunks_count = len(result["ids"])
                            status = "active"
                    except:
                        status = "error"

                all_pdfs.append(
                    {
                        "chat_id": chat["id"],
                        "chat_title": chat.get("title", "Untitled Chat"),
                        "name": chat.get("pdf_name", "Unknown Document"),
                        "chunks": chunks_count,
                        "status": status,
                        "uploaded_at": chat.get("updated_at"),
                    }
                )

        return jsonify({"pdfs": all_pdfs, "total_pdfs": len(all_pdfs)})

    except Exception as e:
        print(f"❌ Error getting all PDFs: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/chats/<chat_id>/debug-search", methods=["POST"])
def debug_search(chat_id):
    """Debug endpoint to see what chunks exist and what gets retrieved"""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        collection = pdf_collections.get(chat_id)
        if not collection:
            return jsonify({"error": "No collection found for this chat"})
        
        all_docs = collection.get()
        total_chunks = len(all_docs["documents"]) if all_docs and "documents" in all_docs else 0
        
        results = pdf_processor.query_collection_for_context(collection, query, n_results=10)
        
        debug_info = {
            "query": query,
            "total_chunks_in_collection": total_chunks,
            "retrieved_chunks": len(results),
            "pdf_engine": "PyMuPDF + pdfplumber",
            "sample_chunks": []
        }
        
        for i, (doc, meta) in enumerate(results[:5]):
            debug_info["sample_chunks"].append({
                "chunk_number": i + 1,
                "page": meta.get("page", "unknown"),
                "type": meta.get("type", "unknown"),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "word_count": len(doc.split())
            })
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Startup ----------
if __name__ == "__main__":
    print("🚀 Starting Gemini Chat API Server with EXACT source references...")
    load_chat_history()
    print("📋 Server configuration:")
    print(f"   - Python version: {os.sys.version}")
    print(f"   - Flask CORS enabled: ✅")
    print(f"   - Gemini API key configured: ✅")
    print(f"   - PDF processing: PyMuPDF (fast) + pdfplumber (tables) {'+ Docling fallback' if pdf_processor.docling_available else ''} ✅")
    print(f"   - Vector DB: ChromaDB (persist_directory=./chroma_db) ✅")
    print(f"   - Embedding model: all-MiniLM-L6-v2 ✅")
    print(f"   - Chat history loaded: {len(chat_history)} chats")
    print("🌐 Starting server on http://localhost:5000")
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
