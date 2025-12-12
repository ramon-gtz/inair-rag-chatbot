"""Streamlit RAG Chatbot application with OpenAI-like UI."""
import streamlit as st
import uuid
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from config import validate_config
from database import (
    init_db, create_conversation, add_message, get_conversation_history,
    get_recent_conversations, get_conversation, update_conversation_title,
    delete_conversation
)
from agent import DocumentSearchAgent
from supabase_utils import get_available_accounts, get_available_document_types, get_document_stats


# Page configuration
st.set_page_config(
    page_title="InAir: RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for OpenAI-like design
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #f7f7f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        border: 1px solid #e5e5e6;
    }
    
    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e5e5e6;
    }
    
    /* Source citation styling */
    .source-card {
        background-color: #f9fafb;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 3px solid #10a37f;
        font-size: 0.875rem;
    }
    
    .source-header {
        font-weight: 600;
        color: #10a37f;
        margin-bottom: 0.25rem;
    }
    
    .source-meta {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid #d1d5db;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9fafb;
    }
    
    /* Conversation list item */
    .conversation-item {
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .conversation-item:hover {
        background-color: #f3f4f6;
    }
    
    .conversation-item.active {
        background-color: #e5e7eb;
        font-weight: 600;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */
    
    /* Loading spinner */
    .stSpinner > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent(_api_key_hash=None):
    """Initialize the document search agent (cached).
    
    Args:
        _api_key_hash: Hash of API key to invalidate cache when key changes
    """
    try:
        validate_config()
        agent = DocumentSearchAgent()
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


def update_agent_filters():
    """Update agent's default filters from session state."""
    if st.session_state.agent:
        # Convert filters to the format expected by agent
        filters = {
            'account_ids': st.session_state.filters['account_ids'] if st.session_state.filters['account_ids'] else None,
            'document_types': st.session_state.filters['document_types'] if st.session_state.filters['document_types'] else None,
            'start_date': st.session_state.filters['start_date'],
            'end_date': st.session_state.filters['end_date']
        }
        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}
        st.session_state.agent.update_default_filters(filters)


def initialize_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Get API key hash to invalidate cache when key changes
    import hashlib
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_hash = hashlib.md5(api_key.encode()).hexdigest() if api_key else ""
    
    if 'agent' not in st.session_state or st.session_state.get('api_key_hash') != api_key_hash:
        # Clear cache if API key changed
        if 'api_key_hash' in st.session_state and st.session_state.get('api_key_hash') != api_key_hash:
            initialize_agent.clear()
        st.session_state.api_key_hash = api_key_hash
        st.session_state.agent = initialize_agent(_api_key_hash=api_key_hash)
    
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'account_ids': [],
            'document_types': [],
            'start_date': None,
            'end_date': None
        }
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []


def load_conversation(conversation_id: int):
    """Load a conversation and its messages."""
    conversation = get_conversation(conversation_id)
    if not conversation:
        return
    
    st.session_state.current_conversation_id = conversation_id
    messages = get_conversation_history(conversation_id)
    
    # Convert to display format
    st.session_state.messages = []
    for msg in messages:
        st.session_state.messages.append({
            'role': msg['role'],
            'content': msg['content'],
            'metadata': msg.get('metadata', {})
        })
    
        # History will be loaded when querying the agent


def create_new_conversation():
    """Create a new conversation."""
    conversation_id = create_conversation(st.session_state.session_id)
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = []
    if st.session_state.agent:
        st.session_state.agent.clear_memory()
    return conversation_id


def generate_conversation_title(first_message: str) -> str:
    """Generate a title from the first message."""
    # Take first 50 characters
    title = first_message[:50]
    if len(first_message) > 50:
        title += "..."
    return title


def display_message(message: Dict, index: int):
    """Display a chat message with proper styling."""
    role = message['role']
    content = message['content']
    metadata = message.get('metadata', {})
    
    if role == 'user':
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Assistant</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        sources = metadata.get('sources', [])
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">Source {i}</div>
                        <div>{source}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and conversation history."""
    with st.sidebar:
        st.title("💬 InAir Chatbot")
        
        # Settings section
        with st.expander("⚙️ Search Filters", expanded=False):
            st.markdown("**Optional Filters** (Leave empty to search all documents)")
            
            # Fetch available options from Airtable and Supabase
            try:
                available_accounts = get_available_accounts()  # Returns List[Tuple[id, name]]
                available_doc_types = get_available_document_types()  # Returns Dict[display_name, actual_value]
                stats = get_document_stats()
                
                # Show stats
                if stats['total_documents'] > 0:
                    st.caption(f"📊 {stats['total_documents']} documents | {stats['total_accounts']} accounts | {stats['total_document_types']} document types")
            except Exception as e:
                st.warning(f"Could not load options: {str(e)}")
                available_accounts = []
                available_doc_types = {}
            
            # Account IDs filter - multiselect
            if available_accounts:
                # Create mapping of company names to IDs
                account_id_to_name = {acc_id: acc_name for acc_id, acc_name in available_accounts}
                account_name_to_id = {acc_name: acc_id for acc_id, acc_name in available_accounts}
                
                # Get currently selected account names for default
                current_account_names = [
                    account_id_to_name.get(acc_id, acc_id) 
                    for acc_id in st.session_state.filters.get('account_ids', [])
                    if acc_id in account_id_to_name
                ]
                
                # Display multiselect with company names
                selected_account_names = st.multiselect(
                    "Filter by Accounts",
                    options=[name for _, name in available_accounts],
                    default=current_account_names,
                    help="Select specific client accounts to search within"
                )
                
                # Convert selected names back to IDs for storage
                st.session_state.filters['account_ids'] = [
                    account_name_to_id[name] for name in selected_account_names
                ]
            else:
                st.info("No accounts available")
                st.session_state.filters['account_ids'] = []
            
            # Document types filter - multiselect
            if available_doc_types:
                # Get currently selected display names for default
                value_to_display = {v: k for k, v in available_doc_types.items()}
                current_display_names = [
                    value_to_display.get(doc_type, doc_type)
                    for doc_type in st.session_state.filters.get('document_types', [])
                    if doc_type in value_to_display
                ]
                
                # Display multiselect with friendly names
                selected_display_names = st.multiselect(
                    "Filter by Document Types",
                    options=list(available_doc_types.keys()),
                    default=current_display_names,
                    help="Select specific document types (meetings, emails, etc.)"
                )
                
                # Convert display names to actual values for storage
                st.session_state.filters['document_types'] = [
                    available_doc_types[display_name] for display_name in selected_display_names
                ]
            else:
                st.info("No document types available")
                st.session_state.filters['document_types'] = []
            
            # Date range filters
            st.markdown("**Date Range** (Optional)")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=None, key="start_date_filter")
                st.session_state.filters['start_date'] = (
                    start_date.isoformat() + "T00:00:00Z" if start_date else None
                )
            with col2:
                end_date = st.date_input("End Date", value=None, key="end_date_filter")
                st.session_state.filters['end_date'] = (
                    end_date.isoformat() + "T23:59:59Z" if end_date else None
                )
            
            # Clear filters button
            if st.button("Clear All Filters", use_container_width=True):
                st.session_state.filters = {
                    'account_ids': [],
                    'document_types': [],
                    'start_date': None,
                    'end_date': None
                }
                update_agent_filters()
                st.rerun()
            
            # Update agent filters when settings change
            update_agent_filters()
        
        st.divider()
        
        # Conversation history
        st.subheader("Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            create_new_conversation()
            st.rerun()
        
        # List recent conversations
        conversations = get_recent_conversations(st.session_state.session_id, limit=20)
        
        if conversations:
            for conv in conversations:
                is_active = conv['id'] == st.session_state.current_conversation_id
                title = conv['title']
                message_count = conv['message_count']
                updated = datetime.fromisoformat(conv['updated_at']).strftime("%m/%d %H:%M")
                
                # Truncate title if too long
                if len(title) > 30:
                    title = title[:27] + "..."
                
                if st.button(
                    f"{title} ({message_count})",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    load_conversation(conv['id'])
                    st.rerun()
        else:
            st.info("No conversations yet. Start a new one!")
        
        st.divider()
        
        # Debug logs section
        with st.expander("🐛 Debug Logs", expanded=False):
            if 'debug_logs' in st.session_state and st.session_state.debug_logs:
                # Show last 50 log lines
                log_lines = st.session_state.debug_logs[-50:]
                log_text = "\n".join(log_lines)
                st.text_area(
                    "Recent Debug Output",
                    value=log_text,
                    height=300,
                    disabled=True,
                    key="debug_log_viewer"
                )
                if st.button("Clear Logs", use_container_width=True):
                    st.session_state.debug_logs = []
                    st.rerun()
            else:
                st.info("No debug logs yet. Debug output will appear here when you make queries.")
        
        st.divider()
        
        # Clear current conversation
        if st.session_state.current_conversation_id:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.clear_memory()
                st.rerun()


def main():
    """Main application function."""
    initialize_session_state()
    
    # Initialize database
    init_db()
    
    # Check if agent is initialized
    if st.session_state.agent is None:
        st.error("Agent initialization failed. Please check your environment variables.")
        st.stop()
    
    # Update agent filters on load
    update_agent_filters()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat area
    st.title("Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Start a conversation by typing a message below!")
        else:
            for i, message in enumerate(st.session_state.messages):
                display_message(message, i)
    
    # Input area
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create input form
    with st.form("chat_input", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        with col1:
            user_input = st.text_input(
                "Type your message...",
                key="user_input",
                placeholder="Ask a question about your documents...",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)
    
    # Handle user input
    if submit_button and user_input:
        # Create conversation if needed
        if st.session_state.current_conversation_id is None:
            conversation_id = create_new_conversation()
            # Generate title from first message
            title = generate_conversation_title(user_input)
            update_conversation_title(conversation_id, title)
        else:
            conversation_id = st.session_state.current_conversation_id
        
        # Add user message to UI
        user_message = {
            'role': 'user',
            'content': user_input,
            'metadata': {}
        }
        st.session_state.messages.append(user_message)
        
        # Save user message to database
        add_message(conversation_id, 'user', user_input)
        
        # Display user message immediately
        st.rerun()
    
    # Process the last user message if it hasn't been responded to
    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
        user_message = st.session_state.messages[-1]['content']
        conversation_id = st.session_state.current_conversation_id
        
        # Show loading state
        with st.spinner("Thinking..."):
            # Capture debug output
            debug_output = io.StringIO()
            
            # Get chat history for agent (excluding the current message)
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                chat_history.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            # Query agent and capture debug output
            with redirect_stdout(debug_output), redirect_stderr(debug_output):
                result = st.session_state.agent.query(user_message, chat_history)
            
            # Store debug logs
            debug_text = debug_output.getvalue()
            if debug_text:
                # Split into lines and add timestamp
                lines = debug_text.strip().split('\n')
                timestamp = datetime.now().strftime("%H:%M:%S")
                for line in lines:
                    if line.strip():  # Only add non-empty lines
                        st.session_state.debug_logs.append(f"[{timestamp}] {line}")
                # Keep only last 200 lines
                st.session_state.debug_logs = st.session_state.debug_logs[-200:]
            
            assistant_response = result.get('output', 'No response generated.')
            sources = result.get('sources', [])
            
            # Convert Document objects to serializable format for database storage
            serializable_sources = []
            for source in sources:
                if hasattr(source, 'page_content'):  # It's a Document object
                    serializable_sources.append({
                        'content': source.page_content[:500] + '...' if len(source.page_content) > 500 else source.page_content,
                        'metadata': source.metadata,
                        'type': 'document'
                    })
                elif isinstance(source, dict):
                    # Already a dictionary, use as-is
                    serializable_sources.append(source)
                else:
                    # String or other type, convert to string
                    serializable_sources.append(str(source))
            
            # For UI, keep original sources (can handle Document objects)
            # For database, use serializable format
            assistant_message = {
                'role': 'assistant',
                'content': assistant_response,
                'metadata': {'sources': sources}  # Keep original for UI display
            }
            st.session_state.messages.append(assistant_message)
            
            # Save assistant message to database with serializable sources
            add_message(
                conversation_id,
                'assistant',
                assistant_response,
                {'sources': serializable_sources}
            )
        
        # Rerun to display the response
        st.rerun()


if __name__ == "__main__":
    main()

