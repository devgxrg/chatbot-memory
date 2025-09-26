import os
import warnings
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import traceback
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Suppress warnings and logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")

# Try to import and configure Gemini
try:
    import google.generativeai as genai
    import re
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("📝 Create a .env file in the backend directory with:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini API configured successfully!")
    
except ImportError as e:
    print(f"❌ ERROR: Failed to import google.generativeai: {e}")
    print("📝 Install required packages: pip install google-generativeai")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: Failed to configure Gemini: {e}")
    exit(1)

# Configure Flask
app = Flask(__name__)
CORS(app)

# Global storage
active_conversations = {}
chat_history = []

def load_chat_history():
    """Load chat history from file"""
    global chat_history
    try:
        with open("chat_history.json", "r") as f:
            chat_history = json.load(f)
        print(f"📁 Loaded {len(chat_history)} chat sessions")
    except FileNotFoundError:
        chat_history = []
        print("📁 No existing chat history found, starting fresh")
    except Exception as e:
        print(f"⚠️  Error loading chat history: {e}")
        chat_history = []

def save_chat_history():
    """Save chat history to file"""
    try:
        with open("chat_history.json", "w") as f:
            json.dump(chat_history, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error saving chat history: {e}")

def load_conversation(chat_id):
    """Load a specific conversation memory"""
    try:
        with open(f"memory_{chat_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"⚠️  Error loading conversation {chat_id}: {e}")
        return []

def save_conversation(chat_id, conversation):
    """Save a specific conversation memory"""
    try:
        with open(f"memory_{chat_id}.json", "w") as f:
            json.dump(conversation, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error saving conversation {chat_id}: {e}")

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chat sessions"""
    try:
        return jsonify(chat_history)
    except Exception as e:
        print(f"❌ Error in get_chats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat session"""
    try:
        chat_id = str(uuid.uuid4())
        new_chat = {
            "id": chat_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        chat_history.insert(0, new_chat)
        active_conversations[chat_id] = []
        save_chat_history()
        save_conversation(chat_id, [])
        
        print(f"✅ Created new chat: {chat_id}")
        return jsonify(new_chat), 201
        
    except Exception as e:
        print(f"❌ Error creating chat: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a specific chat conversation"""
    try:
        conversation = load_conversation(chat_id)
        
        # Convert to frontend format
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

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat session"""
    try:
        global chat_history
        chat_history = [chat for chat in chat_history if chat["id"] != chat_id]
        
        # Remove from active conversations
        if chat_id in active_conversations:
            del active_conversations[chat_id]
        
        # Remove memory file
        try:
            os.remove(f"memory_{chat_id}.json")
        except FileNotFoundError:
            pass
        
        save_chat_history()
        print(f"🗑️  Deleted chat: {chat_id}")
        return jsonify({"message": "Chat deleted successfully"})
        
    except Exception as e:
        print(f"❌ Error deleting chat {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def send_message(chat_id):
    """Send a message to a specific chat"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        print(f"💬 Received message for chat {chat_id}: {user_message}")
        
        # Load or initialize conversation
        if chat_id not in active_conversations:
            active_conversations[chat_id] = load_conversation(chat_id)
        
        conversation = active_conversations[chat_id]
        
        # Handle multiple questions
        questions = re.split(r'(?<=\?)\s+', user_message)
        all_responses = []
        
        for question in questions:
            if question.strip():
                # Add user message to conversation
                conversation.append({"role": "user", "parts": [question.strip()]})
                
                print(f"🤖 Generating response for: {question.strip()}")
                
                # Generate response
                try:
                    response = model.generate_content(conversation)
                    if response.text:
                        bot_response = response.text
                    else:
                        bot_response = "I apologize, but I couldn't generate a response. Please try again."
                        print(f"⚠️  No text in response: {response}")
                        
                except Exception as api_error:
                    print(f"❌ Gemini API Error: {api_error}")
                    bot_response = f"I encountered an error while processing your request: {str(api_error)}"
                
                # Add bot response to conversation
                conversation.append({"role": "model", "parts": [bot_response]})
                all_responses.append(bot_response)
                
                print(f"✅ Generated response: {bot_response[:100]}...")
        
        # Combine responses if multiple questions
        final_response = "\n\n".join(all_responses) if len(all_responses) > 1 else all_responses[0]
        
        # Save conversation
        save_conversation(chat_id, conversation)
        
        # Update chat history
        for chat in chat_history:
            if chat["id"] == chat_id:
                chat["updated_at"] = datetime.now().isoformat()
                chat["message_count"] = len(conversation) // 2
                # Update title with first user message if it's still "New Chat"
                if chat["title"] == "New Chat" and user_message:
                    chat["title"] = user_message[:50] + ("..." if len(user_message) > 50 else "")
                break
        
        save_chat_history()
        
        # Return the response
        response_data = {
            "id": str(uuid.uuid4()),
            "text": final_response,
            "sender": "bot",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📤 Sending response: {final_response[:100]}...")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in send_message: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/chats/<chat_id>/rename', methods=['PUT'])
def rename_chat(chat_id):
    """Rename a chat session"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400
        
        for chat in chat_history:
            if chat["id"] == chat_id:
                chat["title"] = new_title
                chat["updated_at"] = datetime.now().isoformat()
                break
        
        save_chat_history()
        print(f"✏️  Renamed chat {chat_id} to: {new_title}")
        return jsonify({"message": "Chat renamed successfully"})
        
    except Exception as e:
        print(f"❌ Error renaming chat {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "gemini_configured": True,
            "chats_loaded": len(chat_history)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Gemini Chat API Server...")
    load_chat_history()
    print("📋 Server configuration:")
    print(f"   - Python version: {os.sys.version}")
    print(f"   - Flask CORS enabled: ✅")
    print(f"   - Gemini API key configured: ✅")
    print(f"   - Chat history loaded: {len(chat_history)} chats")
    print("🌐 Starting server on http://localhost:5000")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()