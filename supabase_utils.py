"""Utility functions for fetching data from Supabase."""
from typing import List, Dict, Optional
from supabase.client import create_client
from config import SUPABASE_URL, SUPABASE_KEY
import streamlit as st


def get_supabase_client():
    """Get Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_accounts() -> List[str]:
    """Fetch distinct account IDs from Supabase."""
    try:
        client = get_supabase_client()
        # Query distinct account_ids from accounts_documents table
        # Use limit to avoid fetching all rows, then get unique values
        response = client.table("accounts_documents").select("account_id").limit(10000).execute()
        
        if response.data:
            # Extract unique account IDs
            account_ids = list(set([row.get("account_id") for row in response.data if row.get("account_id")]))
            return sorted(account_ids)
        return []
    except Exception as e:
        print(f"Error fetching accounts: {str(e)}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_document_types() -> List[str]:
    """Fetch distinct document types from Supabase."""
    try:
        client = get_supabase_client()
        # Query distinct document_type from accounts_documents table
        response = client.table("accounts_documents").select("document_type").limit(10000).execute()
        
        if response.data:
            # Extract unique document types
            doc_types = list(set([row.get("document_type") for row in response.data if row.get("document_type")]))
            return sorted(doc_types)
        return []
    except Exception as e:
        print(f"Error fetching document types: {str(e)}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_document_stats() -> Dict[str, int]:
    """Get statistics about documents in the database."""
    try:
        client = get_supabase_client()
        # Get total count
        response = client.table("accounts_documents").select("id", count="exact").execute()
        total_count = response.count if hasattr(response, 'count') else len(response.data) if response.data else 0
        
        # Get account and doc type counts (these are cached separately)
        accounts = get_available_accounts()
        doc_types = get_available_document_types()
        
        return {
            "total_documents": total_count,
            "total_accounts": len(accounts),
            "total_document_types": len(doc_types)
        }
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        return {"total_documents": 0, "total_accounts": 0, "total_document_types": 0}

