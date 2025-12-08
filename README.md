# RAG Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that connects to a Supabase vector store and uses OpenAI's reasoning models to answer questions about your documents.

## Features

- **Hybrid Multi-Stage Architecture**: Uses gpt-4o-mini for tool orchestration and o3-mini for deep reasoning
- **Semantic Search**: Multi-stage search with query variations for comprehensive document retrieval
- **Intelligent Document Grouping**: Automatically groups documents by meeting, date, or document ID
- **Per-Meeting Breakdowns**: Analyzes and summarizes multiple meetings with detailed breakdowns
- **Conversation Memory**: Persistent conversation history stored in SQLite
- **Source Citations**: View the documents that informed each response
- **Advanced Filtering**: Filter by account IDs, document types, and date ranges
- **OpenAI-like UI**: Clean, minimalistic interface inspired by OpenAI's chat interface
- **Comprehensive Analysis**: Handles queries requiring analysis of many documents (e.g., "all meetings", "every transcript")

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your:
- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. **Start a Conversation**: Type a question in the input field and press Enter or click Send
2. **View Sources**: Click on the "Sources" expander to see which documents informed the response
3. **Filter Results**: Use the sidebar settings to filter by:
   - Account IDs
   - Document types
   - Date ranges
4. **Manage Conversations**: 
   - Create new conversations using the "New Conversation" button
   - Switch between conversations from the sidebar
   - Clear the current conversation to start fresh

## Architecture

- **Frontend**: Streamlit web interface
- **Agent Orchestrator**: gpt-4o-mini for tool calling and search strategy
- **Reasoning Engine**: o3-mini for deep analysis and summarization
- **Multi-Stage Search**: Performs multiple searches with query variations to maximize recall
- **Vector Store**: Supabase with `match_accounts_documents` function
- **Storage**: SQLite database for conversation history

## Project Structure

```
RAG Chatbot/
├── app.py              # Main Streamlit application
├── agent.py            # LangChain agent with hybrid multi-stage architecture
├── retriever.py        # Custom Supabase retriever with filtering
├── database.py         # SQLite database operations
├── config.py           # Configuration management
├── supabase_utils.py   # Supabase utility functions
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── .streamlit/
    └── config.toml     # Streamlit theme configuration
```

## Notes

- Conversations are stored locally in `conversations.db` (excluded from git)
- The system uses a hybrid approach:
  - **gpt-4o-mini**: Handles tool calling, query analysis, and search orchestration
  - **o3-mini**: Performs deep reasoning, meeting analysis, and breakdown synthesis
- Document embeddings use `text-embedding-3-small` model
- Comprehensive queries retrieve up to 150 documents for thorough analysis
- Standard queries retrieve 20 documents
- The system automatically detects when comprehensive analysis is needed (keywords like "all", "every", "complete")

