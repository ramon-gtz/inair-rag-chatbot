"""Utility functions for fetching data from Supabase and Airtable."""
from typing import List, Dict, Optional, Tuple
from supabase.client import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY,
    AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID,
    AIRTABLE_COMPANY_FIELD_ID, AIRTABLE_VIEW_ID
)
import streamlit as st
import requests


def get_supabase_client():
    """Get Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_accounts() -> List[Tuple[str, str]]:
    """Fetch accounts from Airtable.
    
    Returns:
        List of tuples (account_id, company_name) sorted by company name
    """
    try:
        # Make POST request to Airtable API
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}/listRecords"
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "fields": [AIRTABLE_COMPANY_FIELD_ID],
            "returnFieldsByFieldId": True,
            "view": AIRTABLE_VIEW_ID
        }
        
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract account records
        accounts = []
        if "records" in data:
            for record in data["records"]:
                account_id = record.get("id")
                # The field is returned by field ID, so we need to check the fields dict
                fields = record.get("fields", {})
                # The company name might be under the field ID or "Company Name"
                company_name = fields.get("Company Name") or fields.get(AIRTABLE_COMPANY_FIELD_ID)
                
                if account_id and company_name:
                    accounts.append((account_id, company_name))
        
        # Sort by company name
        accounts.sort(key=lambda x: x[1])
        return accounts
        
    except Exception as e:
        print(f"Error fetching accounts from Airtable: {str(e)}")
        return []


def get_available_document_types() -> Dict[str, str]:
    """Get available document types with display names.
    
    Returns:
        Dictionary mapping display names to actual database values
    """
    # Fixed mapping of display names to actual database values
    return {
        "Meeting Transcript": "meeting transcript",
        "Email": "email",
        "Slack Message": "message"
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_document_stats() -> Dict[str, int]:
    """Get statistics about documents in the database."""
    try:
        client = get_supabase_client()
        # Get total count
        response = client.table("accounts_documents").select("id", count="exact").execute()
        total_count = response.count if hasattr(response, 'count') else len(response.data) if response.data else 0
        
        # Get account and doc type counts (these are cached separately)
        accounts = get_available_accounts()  # Returns List[Tuple[id, name]]
        doc_types = get_available_document_types()  # Returns Dict[display, value]
        
        return {
            "total_documents": total_count,
            "total_accounts": len(accounts),
            "total_document_types": len(doc_types)
        }
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        return {"total_documents": 0, "total_accounts": 0, "total_document_types": 0}

