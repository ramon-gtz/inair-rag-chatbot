"""Configuration management for the RAG Chatbot application."""
import streamlit as st
from pathlib import Path

# OpenAI Configuration
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()  # Strip whitespace/newlines
OPENAI_MODEL = "o3-mini"  # Reasoning model for final analysis
AGENT_MODEL = "gpt-4o-mini"  # Agent model for tool calling and orchestration
EMBEDDING_MODEL = "text-embedding-3-small"

# Multi-Stage Search Configuration
MAX_COMPREHENSIVE_K = 20  # Maximum documents for comprehensive searches
STANDARD_K = 10  # Standard search result count
REFINED_K = 15  # Refined search result count

# Query Performance Settings
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include (0.0-1.0)
RPC_TIMEOUT_SECONDS = 30  # Timeout for Supabase RPC calls
PROGRESSIVE_K_VALUES = [5, 10, 15, 20]  # Progressive k values to try on timeout

# Supabase Configuration
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

# Airtable Configuration
AIRTABLE_API_KEY = st.secrets.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = "appF19Ns3tXbwO9gA"
AIRTABLE_TABLE_ID = "tblM5frQWEQk5sbbn"
AIRTABLE_COMPANY_FIELD_ID = "fldIqx9Er0YbIcGwa"
AIRTABLE_VIEW_ID = "viwLtd3NINE6Xq3Pf"

# Authentication Configuration
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")  # Simple password for app access

# Database Configuration
DB_PATH = Path(__file__).parent / "conversations.db"

# Vector Store Configuration
TABLE_NAME = "accounts_documents"
QUERY_NAME = "match_accounts_documents"
DEFAULT_K = 10  # Default number of documents to retrieve

# Validate required secrets
def validate_config():
    """Validate that all required secrets are set."""
    missing = []
    
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if not AIRTABLE_API_KEY:
        missing.append("AIRTABLE_API_KEY")
    if not APP_PASSWORD:
        missing.append("APP_PASSWORD")
    
    if missing:
        raise ValueError(
            f"Missing required secrets: {', '.join(missing)}\n"
            "Please configure these in .streamlit/secrets.toml file."
        )
    
    return True

