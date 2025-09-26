import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Send, Plus, Trash2, Edit3, MessageSquare, User, Bot, Clock, X, Check, 
  Upload, FileText, AlertCircle, CheckCircle, Loader, Search, Menu
} from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

const ChatApp = () => {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  
  // UI states
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [showDocuments, setShowDocuments] = useState(false);
  const [chatDocs, setChatDocs] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [textareaHeight, setTextareaHeight] = useState(56);
  const [toasts, setToasts] = useState([]);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const dragCounter = useRef(0);

  useEffect(() => {
    loadChats();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Toast notification system
  const addToast = (type, title, message = '') => {
    const id = Date.now();
    const toast = { id, type, title, message, show: true };
    setToasts(prev => [...prev, toast]);
    
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, show: false } : t));
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, 300);
    }, 3000);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, show: false } : t));
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 300);
  };

  const loadChats = async () => {
    try {
      const response = await fetch(`${API_BASE}/chats`);
      const data = await response.json();
      setChats(data);
    } catch (error) {
      console.error('Failed to load chats:', error);
      addToast('error', 'Failed to load conversations');
    }
  };

  const createNewChat = async () => {
    try {
      const response = await fetch(`${API_BASE}/chats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const newChat = await response.json();
      setChats(prev => [newChat, ...prev]);
      setActiveChat(newChat.id);
      setMessages([]);
      setShowDocuments(false);
    } catch (error) {
      console.error('Failed to create chat:', error);
      addToast('error', 'Failed to create new chat');
    }
  };

  const loadChatDocs = async (chatId) => {
    if (!chatId) return;
    try {
      const response = await fetch(`${API_BASE}/chats/${chatId}/pdfs`);
      if (response.ok) {
        const data = await response.json();
        setChatDocs(data.pdfs || []);
      } else {
        setChatDocs([]);
      }
    } catch (error) {
      console.error('Failed to load chat documents:', error);
      setChatDocs([]);
    }
  };

  // Load documents whenever activeChat changes
  useEffect(() => {
    if (activeChat) {
      loadChatDocs(activeChat);
    } else {
      setChatDocs([]);
    }
  }, [activeChat]);

  const deleteDoc = async (chatId, docName) => {
    try {
      const response = await fetch(`${API_BASE}/chats/${chatId}/pdfs/${encodeURIComponent(docName)}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        // Refresh both chats and documents
        await loadChats();
        await loadChatDocs(chatId);
        addToast('success', 'Document removed successfully');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to delete document');
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
      addToast('error', 'Failed to remove document', error.message);
    }
  };

  const loadChat = async (chatId) => {
    if (chatId === activeChat) return;
    
    try {
      setActiveChat(chatId);
      const response = await fetch(`${API_BASE}/chats/${chatId}`);
      const data = await response.json();
      setMessages(data);
      setShowDocuments(false);
      // Documents will be loaded by useEffect when activeChat changes
    } catch (error) {
      console.error('Failed to load chat:', error);
      addToast('error', 'Failed to load conversation');
    }
  };

  const deleteChat = async (chatId, e) => {
    e.stopPropagation();
    if (!confirm('Delete this conversation? This action cannot be undone.')) return;

    try {
      await fetch(`${API_BASE}/chats/${chatId}`, { method: 'DELETE' });
      setChats(prev => prev.filter(chat => chat.id !== chatId));
      if (chatId === activeChat) {
        setActiveChat(null);
        setMessages([]);
        setShowDocuments(false);
      }
      addToast('success', 'Conversation deleted');
    } catch (error) {
      console.error('Failed to delete chat:', error);
      addToast('error', 'Failed to delete conversation');
    }
  };

  const renameChat = async (chatId, newTitle) => {
    try {
      await fetch(`${API_BASE}/chats/${chatId}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
      });
      setChats(prev => prev.map(chat => 
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      ));
      setEditingChatId(null);
      addToast('success', 'Chat renamed');
    } catch (error) {
      console.error('Failed to rename chat:', error);
      addToast('error', 'Failed to rename chat');
    }
  };

  const handleFileUpload = async (file) => {
    if (!activeChat) {
      await createNewChat();
    }

    addToast('info', 'Uploading document...', 'Processing your PDF file');

    try {
      const formData = new FormData();
      formData.append('pdf', file);
      
      const response = await fetch(`${API_BASE}/chats/${activeChat}/upload-pdf`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (response.ok) {
        addToast('success', 'Document uploaded!', `${file.name} is ready for questions`);
        // Refresh both chats and documents
        await loadChats();
        if (activeChat) {
          await loadChatDocs(activeChat);
        }
      } else {
        throw new Error(result.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      addToast('error', 'Upload failed', error.message);
    }
  };

  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
    
    // Auto-expand textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 120);
    textarea.style.height = newHeight + 'px';
    setTextareaHeight(newHeight);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragOver(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    dragCounter.current = 0;
    
    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(file => file.type === 'application/pdf');
    
    if (pdfFile) {
      handleFileUpload(pdfFile);
    } else {
      addToast('error', 'Invalid file type', 'Please upload a PDF file');
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;
    
    let currentChatId = activeChat;
    if (!currentChatId) {
      await createNewChat();
      currentChatId = activeChat;
    }

    const userMessage = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentMessage = inputMessage;
    setInputMessage('');
    setTextareaHeight(56); // Reset textarea height
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chats/${currentChatId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: currentMessage })
      });

      if (response.ok) {
        const botResponse = await response.json();
        setMessages(prev => [...prev, {
          ...botResponse,
          hasContext: botResponse.used_pdf_context
        }]);
        loadChats();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to send message');
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        text: `Sorry, I encountered an error. Please try again.`,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      }]);
      addToast('error', 'Failed to send message', error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    if (e.key === 'Escape') {
      setInputMessage('');
      setTextareaHeight(56);
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const getCurrentChat = () => chats.find(c => c.id === activeChat);

  // Toast component
  const Toast = ({ toast, onRemove }) => (
    <div className={`fixed top-4 right-4 max-w-sm w-full transform transition-all duration-300 ease-out ${
      toast.show ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
    } z-50`}>
      <div className={`rounded-xl shadow-lg border-l-4 p-4 backdrop-blur-sm ${
        toast.type === 'success' 
          ? 'bg-white/95 border-green-500 text-gray-800' 
          : toast.type === 'error'
            ? 'bg-white/95 border-red-500 text-gray-800'
            : 'bg-white/95 border-blue-500 text-gray-800'
      }`}>
        <div className="flex items-start gap-3">
          <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
            toast.type === 'success' ? 'bg-green-100' : 
            toast.type === 'error' ? 'bg-red-100' : 'bg-blue-100'
          }`}>
            {toast.type === 'success' && <CheckCircle className="w-4 h-4 text-green-600" />}
            {toast.type === 'error' && <AlertCircle className="w-4 h-4 text-red-600" />}
            {toast.type === 'info' && <Loader className="w-4 h-4 text-blue-600 animate-spin" />}
          </div>
          
          <div className="flex-1">
            <p className="font-medium text-sm">{toast.title}</p>
            {toast.message && <p className="text-xs text-gray-600 mt-1">{toast.message}</p>}
          </div>
          
          <button 
            onClick={() => onRemove(toast.id)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-white">
      {/* Toast notifications */}
      {toasts.map(toast => (
        <Toast key={toast.id} toast={toast} onRemove={removeToast} />
      ))}

      {/* Enhanced Drag & Drop Overlay */}
      {isDragOver && (
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-md z-50 flex items-center justify-center animate-in fade-in duration-200">
          <div className="bg-white/90 backdrop-blur-sm rounded-3xl p-8 shadow-2xl border-2 border-dashed border-blue-300 max-w-sm mx-auto transform scale-110">
            <div className="relative">
              <Upload className="w-20 h-20 text-blue-500 mx-auto mb-4 animate-bounce" />
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                <Check className="w-4 h-4 text-white" />
              </div>
            </div>
            <h3 className="text-xl font-bold text-center text-gray-800 mb-2">
              Drop your PDF here
            </h3>
            <p className="text-gray-600 text-center text-sm">
              We'll analyze it and make it searchable
            </p>
          </div>
        </div>
      )}

      {/* Enhanced Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-gray-50 border-r border-gray-200 flex flex-col overflow-hidden`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={createNewChat}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl transition-all duration-200 font-medium shadow-sm hover:shadow-md transform hover:-translate-y-0.5"
          >
            <Plus className="w-5 h-5" />
            New Chat
          </button>
        </div>

        {/* Enhanced Chat List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {chats.map((chat) => (
            <div
              key={chat.id}
              onClick={() => loadChat(chat.id)}
              className={`group relative p-3 rounded-xl cursor-pointer transition-all duration-200 ${
                activeChat === chat.id
                  ? 'bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 shadow-sm'
                  : 'hover:bg-gray-100 hover:shadow-sm'
              }`}
            >
              {/* Status indicator */}
              <div className={`absolute left-1 top-1/2 transform -translate-y-1/2 w-1 h-8 rounded-full transition-all duration-200 ${
                activeChat === chat.id ? 'bg-gradient-to-b from-blue-500 to-purple-500' : 'bg-transparent'
              }`} />
              
              <div className="flex items-start gap-3 ml-2">
                {/* Enhanced Chat avatar */}
                <div className="relative flex-shrink-0">
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                    chat.has_pdf 
                      ? 'bg-gradient-to-br from-purple-500 to-blue-500 text-white shadow-md' 
                      : 'bg-gray-100 text-gray-600'
                  }`}>
                    {chat.has_pdf ? <FileText className="w-5 h-5" /> : <MessageSquare className="w-5 h-5" />}
                  </div>
                  
                  {/* Active indicator */}
                  {activeChat === chat.id && (
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 border-2 border-white rounded-full animate-pulse" />
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  {editingChatId === chat.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter') {
                            renameChat(chat.id, editTitle);
                          } else if (e.key === 'Escape') {
                            setEditingChatId(null);
                          }
                        }}
                        className="flex-1 bg-white border border-gray-300 rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
                        autoFocus
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          renameChat(chat.id, editTitle);
                        }}
                        className="text-green-500 hover:text-green-600 transition-colors"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingChatId(null);
                        }}
                        className="text-gray-400 hover:text-gray-500 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-gray-800 truncate text-sm">
                          {chat.title}
                        </h3>
                        {chat.has_pdf && (
                          <span className="bg-gradient-to-r from-purple-100 to-blue-100 text-purple-700 px-2 py-0.5 rounded-full text-xs font-medium">
                            {chat.pdf_count} doc{chat.pdf_count > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <Clock className="w-3 h-3" />
                        <span>{formatDate(chat.updated_at)}</span>
                        <div className="w-1 h-1 bg-gray-300 rounded-full" />
                        <span>{chat.message_count} msgs</span>
                      </div>
                    </>
                  )}
                </div>

                {/* Enhanced hover actions */}
                {editingChatId !== chat.id && (
                  <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingChatId(chat.id);
                        setEditTitle(chat.title);
                      }}
                      className="p-1.5 hover:bg-blue-100 rounded-lg transition-colors"
                      title="Rename"
                    >
                      <Edit3 className="w-3.5 h-3.5 text-gray-600" />
                    </button>
                    <button 
                      onClick={(e) => deleteChat(chat.id, e)}
                      className="p-1.5 hover:bg-red-100 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-gray-600" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {chats.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs text-gray-400 mt-1">Create your first chat to get started</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div 
        className="flex-1 flex flex-col"
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {activeChat ? (
          <>
            {/* Enhanced Chat Header */}
            <div className="bg-white border-b border-gray-200 p-4 sticky top-0 backdrop-blur-md bg-white/90 z-40">
              <div className="flex items-center justify-between max-w-4xl mx-auto">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <button
                    onClick={() => setSidebarOpen(!sidebarOpen)}
                    className="lg:hidden p-2 hover:bg-gray-100 rounded-xl transition-colors"
                  >
                    <Menu className="w-5 h-5 text-gray-600" />
                  </button>
                  
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center shadow-md">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  
                  <div className="min-w-0 flex-1">
                    <h1 className="font-bold text-gray-800 truncate text-lg">
                      {getCurrentChat()?.title || 'New Chat'}
                    </h1>
                    <p className="text-sm text-gray-500 truncate">
                      AI Assistant
                      {getCurrentChat()?.has_pdf && (
                        <span className="ml-1 hidden sm:inline">• Document loaded</span>
                      )}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 flex-shrink-0">
                  {getCurrentChat()?.has_pdf && (
                    <button
                      onClick={() => setShowDocuments(!showDocuments)}
                      className="hidden sm:flex items-center gap-2 px-3 py-2 text-sm bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-xl transition-colors shadow-sm"
                    >
                      <FileText className="w-4 h-4" />
                      <span>Documents</span>
                    </button>
                  )}
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl transition-colors shadow-sm"
                  >
                    <Upload className="w-4 h-4" />
                    <span className="hidden sm:inline">Upload</span>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) handleFileUpload(file);
                      e.target.value = '';
                    }}
                    className="hidden"
                  />
                </div>
              </div>
              
              {/* Enhanced Documents Panel */}
              {showDocuments && getCurrentChat()?.has_pdf && (
                <div className="max-w-4xl mx-auto mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl border border-blue-200 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-gray-800 flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-600" />
                      Your Documents ({chatDocs.length})
                    </h3>
                    <button
                      onClick={() => setShowDocuments(false)}
                      className="text-gray-400 hover:text-gray-600 p-1 rounded-lg hover:bg-white/50 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  
                  {/* Loading state */}
                  {chatDocs.length === 0 && getCurrentChat()?.has_pdf && (
                    <div className="flex items-center justify-center py-4">
                      <Loader className="w-5 h-5 animate-spin text-blue-500 mr-2" />
                      <span className="text-sm text-gray-600">Loading documents...</span>
                    </div>
                  )}
                  
                  {chatDocs.length > 0 ? (
                    <div className="space-y-3">
                      {chatDocs.map((doc, index) => (
                        <div key={index} className="flex items-center justify-between p-4 bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all duration-200">
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <div className="p-2 bg-blue-100 rounded-lg flex-shrink-0">
                              <FileText className="w-5 h-5 text-blue-600" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-gray-800 text-sm truncate" title={doc.name}>
                                {doc.name}
                              </p>
                              <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                                <span>{doc.chunks || 0} sections indexed</span>
                                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full ${
                                  doc.status === 'active' 
                                    ? 'bg-green-100 text-green-700' 
                                    : 'bg-red-100 text-red-700'
                                }`}>
                                  <div className={`w-1.5 h-1.5 rounded-full ${
                                    doc.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                                  }`} />
                                  {doc.status === 'active' ? 'Ready' : 'Inactive'}
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <button
                              onClick={() => {
                                if (confirm(`Remove "${doc.name}" from this chat?\n\nThis will permanently delete the document and all its indexed content.`)) {
                                  deleteDoc(activeChat, doc.name);
                                }
                              }}
                              className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                              title="Remove document"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      ))}
                      
                      {/* Document usage info */}
                      <div className="mt-4 p-3 bg-blue-100/50 rounded-lg border border-blue-200/50">
                        <div className="flex items-start gap-2">
                          <Search className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <div className="text-xs text-blue-800">
                            <p className="font-medium mb-1">How it works:</p>
                            <p>When you ask questions, I'll search through your uploaded documents to find relevant information and include it in my responses. Look for the purple chat bubbles to see when I'm using your document content.</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : !getCurrentChat()?.has_pdf ? (
                    <div className="text-center py-6 text-gray-500">
                      <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm font-medium mb-1">No documents uploaded yet</p>
                      <p className="text-xs text-gray-400">Upload a PDF to start chatting with your documents</p>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="mt-3 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-xs transition-colors"
                      >
                        Upload Document
                      </button>
                    </div>
                  ) : null}
                </div>
              )}
            </div>

            {/* Enhanced Messages */}
            <div className="flex-1 overflow-y-auto bg-gray-50 scroll-smooth">
              <div className="max-w-4xl mx-auto p-6 space-y-6">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-4 animate-in slide-in-from-left duration-300 ${
                      message.sender === 'user' ? 'flex-row-reverse' : ''
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${
                      message.sender === 'user' 
                        ? 'bg-gradient-to-br from-green-500 to-emerald-500' 
                        : message.hasContext
                          ? 'bg-gradient-to-br from-purple-500 to-blue-500'
                          : 'bg-gradient-to-br from-gray-600 to-gray-700'
                    }`}>
                      {message.sender === 'user' ? (
                        <User className="w-5 h-5 text-white" />
                      ) : (
                        <Bot className="w-5 h-5 text-white" />
                      )}
                    </div>
                    
                    <div className={`flex-1 max-w-2xl ${
                      message.sender === 'user' ? 'text-right' : ''
                    }`}>
                      {/* Enhanced message bubble */}
                      <div className={`inline-block p-4 rounded-2xl shadow-sm transition-all duration-200 hover:shadow-md ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-blue-500/20'
                          : message.isError
                            ? 'bg-red-50 border-2 border-red-200 text-red-800 shadow-red-500/10'
                            : message.hasContext
                              ? 'bg-white border-2 border-purple-200 text-gray-800 shadow-purple-500/10'
                              : 'bg-white border border-gray-200 text-gray-800 shadow-gray-500/10'
                      }`}>
                        <div className={`prose prose-sm max-w-none leading-relaxed ${
                            message.sender === 'user' 
                                ? 'prose-invert' // White text for user messages
                                : message.hasContext 
                                ? 'prose-purple' // Purple theme for context messages
                                : 'prose-gray' // Default gray theme
                            }`}>
                            <ReactMarkdown 
                                remarkPlugins={[remarkGfm]}
                                components={{
                                // Custom styling for markdown elements
                                h1: ({children}) => <h1 className="text-lg font-bold mb-2 text-gray-800">{children}</h1>,
                                h2: ({children}) => <h2 className="text-base font-bold mb-2 text-gray-800">{children}</h2>,
                                h3: ({children}) => <h3 className="text-sm font-bold mb-1 text-gray-800">{children}</h3>,
                                strong: ({children}) => <strong className="font-bold text-gray-900">{children}</strong>,
                                em: ({children}) => <em className="italic text-gray-800">{children}</em>,
                                ul: ({children}) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                                ol: ({children}) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                                li: ({children}) => <li className="text-gray-800">{children}</li>,
                                p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>,
                                code: ({node, inline, className, children, ...props}) => {
                                    return inline ? (
                                    <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800" {...props}>
                                        {children}
                                    </code>
                                    ) : (
                                    <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto mb-3">
                                        <code className="text-sm font-mono text-gray-800" {...props}>
                                        {children}
                                        </code>
                                    </pre>
                                    );
                                },
                                table: ({children}) => (
                                    <div className="overflow-x-auto mb-3">
                                    <table className="min-w-full border border-gray-300 rounded-lg">
                                        {children}
                                    </table>
                                    </div>
                                ),
                                thead: ({children}) => <thead className="bg-gray-50">{children}</thead>,
                                tbody: ({children}) => <tbody>{children}</tbody>,
                                tr: ({children}) => <tr className="border-b border-gray-200">{children}</tr>,
                                th: ({children}) => (
                                    <th className="px-4 py-2 text-left font-semibold text-gray-800 border-r border-gray-300 last:border-r-0">
                                    {children}
                                    </th>
                                ),
                                td: ({children}) => (
                                    <td className="px-4 py-2 text-gray-700 border-r border-gray-300 last:border-r-0">
                                    {children}
                                    </td>
                                ),
                                blockquote: ({children}) => (
                                    <blockquote className="border-l-4 border-blue-500 pl-4 py-2 mb-3 bg-blue-50 rounded-r">
                                    {children}
                                    </blockquote>
                                ),
                                a: ({href, children}) => (
                                    <a 
                                    href={href} 
                                    className="text-blue-600 hover:text-blue-800 underline" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    >
                                    {children}
                                    </a>
                                ),
                                }}
                            >
                                {message.text}
                            </ReactMarkdown>
                            </div>
                        
                        {/* Enhanced status indicator for user messages */}
                        {message.sender === 'user' && (
                          <div className="flex justify-end mt-2">
                            <div className="w-4 h-4 rounded-full bg-white/20 flex items-center justify-center">
                              <Check className="w-2.5 h-2.5 text-white" />
                            </div>
                          </div>
                        )}
                        
                        {/* Enhanced context indicator */}
                        {message.hasContext && (
                          <div className="mt-3 pt-3 border-t border-purple-200/50">
                            <div className="flex items-center gap-2 text-xs text-purple-700 bg-purple-50 px-3 py-1.5 rounded-full w-fit">
                              <FileText className="w-3 h-3" />
                              <span>Enhanced with your document</span>
                            </div>
                          </div>
                        )}
                      </div>
                      
                      <div className={`text-xs text-gray-500 mt-2 ${
                        message.sender === 'user' ? 'text-right' : ''
                      }`}>
                        {formatTime(message.timestamp)}
                        {message.hasContext && (
                          <span className="ml-2 text-purple-500 font-medium">• Document Context</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Enhanced loading animation */}
                {isLoading && (
                  <div className="flex gap-4 animate-in slide-in-from-left duration-300">
                    <div className="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-700 rounded-xl flex items-center justify-center shadow-md">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                    
                    <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm max-w-xs">
                      <div className="flex items-center gap-3">
                        {/* Enhanced thinking animation */}
                        <div className="flex gap-1">
                          {[0, 1, 2].map((i) => (
                            <div
                              key={i}
                              className="w-2 h-2 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full animate-bounce"
                              style={{ animationDelay: `${i * 0.1}s` }}
                            />
                          ))}
                        </div>
                        
                        <div className="text-sm text-gray-600">
                          {getCurrentChat()?.has_pdf ? (
                            <span className="flex items-center gap-2">
                              <Search className="w-4 h-4 animate-spin" />
                              Searching documents...
                            </span>
                          ) : (
                            'AI is thinking...'
                          )}
                        </div>
                      </div>
                      
                      {/* Progress bar for document search */}
                      {getCurrentChat()?.has_pdf && (
                        <div className="mt-3 w-full bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-1.5 rounded-full animate-pulse transition-all duration-1000" 
                            style={{ width: '60%' }} 
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Enhanced Message Input */}
            <div className="bg-white border-t border-gray-200 p-4 sticky bottom-0">
              <div className="max-w-4xl mx-auto flex gap-3">
                <div className="flex-1 relative">
                  <textarea
                    value={inputMessage}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask me anything... (Shift+Enter for new line, Esc to clear)"
                    className="w-full px-4 py-3 pr-14 bg-gray-50 border-2 border-gray-200 rounded-2xl resize-none focus:outline-none focus:ring-3 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 placeholder-gray-500 shadow-sm"
                    style={{ height: textareaHeight }}
                    disabled={isLoading}
                  />
                  
                  {/* Character counter */}
                  {inputMessage.length > 500 && (
                    <div className="absolute bottom-2 left-4 text-xs text-gray-400 bg-white/80 backdrop-blur-sm px-2 py-1 rounded-full">
                      {inputMessage.length}/2000
                    </div>
                  )}
                  
                  {/* Enhanced send button */}
                  <button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isLoading}
                    className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2.5 rounded-xl transition-all duration-200 ${
                      inputMessage.trim() && !isLoading
                        ? 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg hover:shadow-blue-500/25 scale-100 hover:scale-105'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed scale-95'
                    }`}
                  >
                    {isLoading ? (
                      <Loader className="w-5 h-5 animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        ) : (
          /* Enhanced Welcome Screen */
          <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50">
            <div className="text-center max-w-lg mx-auto p-8">
              <div className="relative mb-8">
                <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-500 rounded-3xl flex items-center justify-center mx-auto shadow-2xl shadow-blue-500/25">
                  <Bot className="w-12 h-12 text-white" />
                </div>
                <div className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center animate-pulse">
                  <div className="w-3 h-3 bg-white rounded-full" />
                </div>
              </div>
              
              <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Chat with AI
              </h1>
              
              <p className="text-gray-600 mb-8 text-lg leading-relaxed">
                Start a conversation or upload a document to chat with its content. 
                Get instant, intelligent responses powered by advanced AI.
              </p>
              
              <button
                onClick={createNewChat}
                className="group px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-2xl font-semibold text-lg transition-all duration-200 shadow-xl hover:shadow-2xl transform hover:-translate-y-1 hover:scale-105"
              >
                <span className="flex items-center gap-3">
                  Start Chatting
                  <div className="w-0 group-hover:w-5 transition-all duration-200 overflow-hidden">
                    <Send className="w-5 h-5" />
                  </div>
                </span>
              </button>
              
              <div className="mt-8 flex items-center justify-center gap-6 text-sm text-gray-500">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span>Online</span>
                </div>
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  <span>PDF Support</span>
                </div>
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" />
                  <span>Multi-Chat</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatApp;