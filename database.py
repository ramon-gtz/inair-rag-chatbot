"""SQLite database operations for conversation history."""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from config import DB_PATH


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_session_id 
        ON conversations(session_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
        ON messages(conversation_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_created_at 
        ON messages(created_at)
    """)
    
    conn.commit()
    conn.close()


def create_conversation(session_id: str, title: Optional[str] = None) -> int:
    """Create a new conversation and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversations (session_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, title, datetime.now(), datetime.now()))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return conversation_id


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    metadata: Optional[Dict] = None
) -> int:
    """Add a message to a conversation and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (conversation_id, role, content, metadata_json, datetime.now()))
    
    # Update conversation's updated_at timestamp
    cursor.execute("""
        UPDATE conversations 
        SET updated_at = ? 
        WHERE id = ?
    """, (datetime.now(), conversation_id))
    
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return message_id


def get_conversation_history(conversation_id: int) -> List[Dict]:
    """Retrieve all messages for a conversation in chronological order."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content, metadata, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, (conversation_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        messages.append({
            'role': row['role'],
            'content': row['content'],
            'metadata': metadata,
            'created_at': row['created_at']
        })
    
    return messages


def get_recent_conversations(session_id: str, limit: int = 20) -> List[Dict]:
    """Get recent conversations for a session."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, created_at, updated_at,
               (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) as message_count
        FROM conversations
        WHERE session_id = ?
        ORDER BY updated_at DESC
        LIMIT ?
    """, (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    conversations = []
    for row in rows:
        conversations.append({
            'id': row['id'],
            'title': row['title'] or f"Conversation {row['id']}",
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'message_count': row['message_count']
        })
    
    return conversations


def get_conversation(conversation_id: int) -> Optional[Dict]:
    """Get a single conversation by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, session_id, title, created_at, updated_at
        FROM conversations
        WHERE id = ?
    """, (conversation_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        'id': row['id'],
        'session_id': row['session_id'],
        'title': row['title'],
        'created_at': row['created_at'],
        'updated_at': row['updated_at']
    }


def update_conversation_title(conversation_id: int, title: str):
    """Update the title of a conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE conversations 
        SET title = ?, updated_at = ?
        WHERE id = ?
    """, (title, datetime.now(), conversation_id))
    
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    
    conn.commit()
    conn.close()


def get_message_count(conversation_id: int) -> int:
    """Get the number of messages in a conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM messages
        WHERE conversation_id = ?
    """, (conversation_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return row['count'] if row else 0

