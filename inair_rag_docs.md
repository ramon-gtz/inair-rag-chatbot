# Vector Store Document Search - LangChain Integration Guide

## Overview

This document explains how to use the `match_accounts_documents` PostgreSQL function with LangChain to build AI agents that can retrieve and reason over documents stored in a Supabase vector store.

## Architecture

```
User Query → LangChain Agent → Embedding Model → Vector Search → Retrieved Documents → LLM Response
```

## Setup

### Installation

```bash
pip install langchain langchain-openai langchain-community supabase
```

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
```

## Function Overview

The underlying PostgreSQL function accepts:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query_embedding` | vector | Embedding vector from your query |
| `match_count` | integer | Number of results to return |
| `filter` | jsonb | Filter criteria (account_ids, document_types, etc.) |
| `start_date` | text | Optional - ISO 8601 date string |
| `end_date` | text | Optional - ISO 8601 date string |

**Supported Filters:**
- `account_id` or `account_ids` (array)
- `document_id` or `document_ids` (array)
- `document_type` or `document_types` (array)

## LangChain Implementation

### 1. Basic Setup with SupabaseVectorStore

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
import os

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="accounts_documents",
    query_name="match_accounts_documents"
)
```

### 2. Create a Custom Retriever with Filters

```python
from langchain.schema import BaseRetriever, Document
from typing import List, Optional, Dict, Any
from pydantic import Field

class FilteredSupabaseRetriever(BaseRetriever):
    """Custom retriever with support for dynamic filters and date ranges."""
    
    vector_store: SupabaseVectorStore = Field(...)
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on query and filters."""
        # Build filter object
        filter_obj = {}
        
        # Add array filters if provided
        if "account_ids" in self.search_kwargs:
            filter_obj["account_ids"] = self.search_kwargs["account_ids"]
        
        if "document_types" in self.search_kwargs:
            filter_obj["document_types"] = self.search_kwargs["document_types"]
        
        if "document_ids" in self.search_kwargs:
            filter_obj["document_ids"] = self.search_kwargs["document_ids"]
        
        # Single value filters (backward compatible)
        if "account_id" in self.search_kwargs:
            filter_obj["account_id"] = self.search_kwargs["account_id"]
        
        # Get k (number of results)
        k = self.search_kwargs.get("k", 10)
        
        # Call the custom match function
        results = self.vector_store.client.rpc(
            'match_accounts_documents',
            {
                'query_embedding': self.vector_store.embedding.embed_query(query),
                'match_count': k,
                'filter': filter_obj,
                'start_date': self.search_kwargs.get('start_date'),
                'end_date': self.search_kwargs.get('end_date')
            }
        ).execute()
        
        # Convert to LangChain documents
        documents = []
        for result in results.data:
            doc = Document(
                page_content=result['content'],
                metadata={
                    **result['metadata'],
                    'similarity': result['similarity'],
                    'id': result['id']
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of retrieval."""
        return self._get_relevant_documents(query)
```

### 3. Usage Examples

#### Example 1: Simple Retrieval

```python
# Create retriever
retriever = FilteredSupabaseRetriever(
    vector_store=vector_store,
    search_kwargs={
        "k": 10  # Return top 10 results
    }
)

# Retrieve documents
docs = retriever.get_relevant_documents("What were the key decisions from the Q4 planning meeting?")

for doc in docs:
    print(f"Similarity: {doc.metadata['similarity']}")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Account: {doc.metadata.get('account_id')}")
    print("---")
```

#### Example 2: Filter by Account

```python
retriever = FilteredSupabaseRetriever(
    vector_store=vector_store,
    search_kwargs={
        "k": 15,
        "account_id": "acc_123"
    }
)

docs = retriever.get_relevant_documents("Show me all customer feedback")
```

#### Example 3: Multiple Accounts and Document Types

```python
retriever = FilteredSupabaseRetriever(
    vector_store=vector_store,
    search_kwargs={
        "k": 20,
        "account_ids": ["acc_123", "acc_456", "acc_789"],
        "document_types": ["meeting_notes", "email", "transcript"]
    }
)

docs = retriever.get_relevant_documents("What are the main challenges mentioned?")
```

#### Example 4: Date Range Filter

```python
retriever = FilteredSupabaseRetriever(
    vector_store=vector_store,
    search_kwargs={
        "k": 25,
        "account_ids": ["acc_123", "acc_456"],
        "start_date": "2024-11-01T00:00:00Z",
        "end_date": "2024-12-01T23:59:59Z"
    }
)

docs = retriever.get_relevant_documents("Summarize recent product discussions")
```

### 4. RAG Chain with RetrievalQA

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create retriever with filters
retriever = FilteredSupabaseRetriever(
    vector_store=vector_store,
    search_kwargs={
        "k": 10,
        "account_ids": ["acc_123"],
        "document_types": ["meeting_notes"]
    }
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask questions
result = qa_chain({"query": "What were the action items from the last sprint planning?"})

print("Answer:", result['result'])
print("\nSources:")
for doc in result['source_documents']:
    print(f"- {doc.metadata.get('document_type')}: {doc.metadata.get('document_id')} (similarity: {doc.metadata.get('similarity'):.3f})")
```

### 5. Conversational Retrieval Chain

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Multi-turn conversation
response1 = qa_chain({"question": "What were the main topics in our Q4 meetings?"})
print(response1['answer'])

response2 = qa_chain({"question": "What action items came out of those?"})
print(response2['answer'])
```

### 6. LangChain Agent with Custom Tool

```python
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field as PydanticField

# Define input schema
class SearchInput(BaseModel):
    query: str = PydanticField(description="The search query")
    account_ids: Optional[List[str]] = PydanticField(
        default=None, 
        description="List of account IDs to filter by"
    )
    document_types: Optional[List[str]] = PydanticField(
        default=None,
        description="List of document types to filter by (e.g., 'meeting_notes', 'email', 'transcript')"
    )
    start_date: Optional[str] = PydanticField(
        default=None,
        description="Start date in ISO format (e.g., '2024-11-01T00:00:00Z')"
    )
    end_date: Optional[str] = PydanticField(
        default=None,
        description="End date in ISO format (e.g., '2024-12-01T23:59:59Z')"
    )

def search_documents(
    query: str,
    account_ids: Optional[List[str]] = None,
    document_types: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """Search documents in the vector store with optional filters."""
    
    # Build search kwargs
    search_kwargs = {"k": 10}
    
    if account_ids:
        search_kwargs["account_ids"] = account_ids
    if document_types:
        search_kwargs["document_types"] = document_types
    if start_date:
        search_kwargs["start_date"] = start_date
    if end_date:
        search_kwargs["end_date"] = end_date
    
    # Create retriever
    retriever = FilteredSupabaseRetriever(
        vector_store=vector_store,
        search_kwargs=search_kwargs
    )
    
    # Get documents
    docs = retriever.get_relevant_documents(query)
    
    # Format results
    if not docs:
        return "No relevant documents found."
    
    results = []
    for i, doc in enumerate(docs[:5], 1):  # Return top 5
        results.append(
            f"Document {i} (similarity: {doc.metadata.get('similarity', 0):.3f}):\n"
            f"Type: {doc.metadata.get('document_type', 'unknown')}\n"
            f"Account: {doc.metadata.get('account_id', 'unknown')}\n"
            f"Content: {doc.page_content[:300]}...\n"
        )
    
    return "\n---\n".join(results)

# Create tool
search_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_documents",
    description="Search through company documents using semantic search. Can filter by accounts, document types, and date ranges.",
    args_schema=SearchInput
)

# Create agent
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can search through company documents.
    When searching, consider:
    - What accounts are relevant to the query
    - What document types would contain the information
    - If a time range is mentioned, include appropriate date filters
    
    Always provide clear, concise answers based on the retrieved documents."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use the agent
result = agent_executor.invoke({
    "input": "What were the key decisions from TechCorp's meetings in November 2024?"
})

print(result['output'])
```

### 7. Dynamic Filter Configuration

```python
def create_dynamic_retriever(
    account_ids: Optional[List[str]] = None,
    document_types: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    k: int = 10
) -> FilteredSupabaseRetriever:
    """Factory function to create retrievers with different configurations."""
    
    search_kwargs = {"k": k}
    
    if account_ids:
        search_kwargs["account_ids"] = account_ids
    if document_types:
        search_kwargs["document_types"] = document_types
    if start_date:
        search_kwargs["start_date"] = start_date
    if end_date:
        search_kwargs["end_date"] = end_date
    
    return FilteredSupabaseRetriever(
        vector_store=vector_store,
        search_kwargs=search_kwargs
    )

# Usage
retriever = create_dynamic_retriever(
    account_ids=["acc_123", "acc_456"],
    document_types=["meeting_notes", "email"],
    start_date="2024-11-01T00:00:00Z",
    k=15
)
```

## Best Practices

### 1. Similarity Threshold Filtering

```python
def filter_by_similarity(docs: List[Document], threshold: float = 0.7) -> List[Document]:
    """Filter documents by similarity score."""
    return [doc for doc in docs if doc.metadata.get('similarity', 0) >= threshold]

# Usage
docs = retriever.get_relevant_documents("query")
high_quality_docs = filter_by_similarity(docs, threshold=0.75)
```

### 2. Error Handling

```python
from supabase.lib.client_options import ClientOptions
import logging

logger = logging.getLogger(__name__)

def safe_retrieval(retriever: FilteredSupabaseRetriever, query: str) -> List[Document]:
    """Retrieve documents with error handling."""
    try:
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []
```

### 3. Metadata Enrichment

```python
def enrich_documents(docs: List[Document]) -> List[Document]:
    """Add additional context to document metadata."""
    for doc in docs:
        # Add formatted date
        if 'created_at' in doc.metadata:
            from datetime import datetime
            doc.metadata['created_at_formatted'] = datetime.fromisoformat(
                doc.metadata['created_at'].replace('Z', '+00:00')
            ).strftime('%Y-%m-%d %H:%M')
        
        # Add quality indicator
        similarity = doc.metadata.get('similarity', 0)
        if similarity >= 0.8:
            doc.metadata['quality'] = 'high'
        elif similarity >= 0.6:
            doc.metadata['quality'] = 'medium'
        else:
            doc.metadata['quality'] = 'low'
    
    return docs
```

## Performance Tips

1. **Start with specific filters** - Narrow the search space with account_ids and document_types
2. **Adjust k based on use case** - Use 5-10 for focused queries, 20-30 for exploratory
3. **Use date filters** - When temporal context matters, add start_date/end_date
4. **Cache embeddings** - LangChain caches embeddings by default, but ensure it's enabled
5. **Monitor similarity scores** - Set thresholds (0.7+) for high-quality matches
6. **Batch processing** - For multiple queries, consider parallel execution

## Troubleshooting

**Issue: No documents returned**
```python
# Debug retrieval
docs = retriever.get_relevant_documents("test query")
print(f"Found {len(docs)} documents")
if not docs:
    # Try without filters
    basic_retriever = FilteredSupabaseRetriever(
        vector_store=vector_store,
        search_kwargs={"k": 5}
    )
    docs = basic_retriever.get_relevant_documents("test query")
    print(f"Without filters: {len(docs)} documents")
```

**Issue: Low similarity scores**
- Check embedding model consistency (use same model for indexing and querying)
- Verify document content quality
- Consider query reformulation

**Issue: Timeout errors**
- Reduce `k` value
- Add more specific filters
- Check database indexes

## Complete Example: Production-Ready Agent

```python
import os
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from supabase.client import create_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSearchAgent:
    def __init__(self):
        # Initialize clients
        self.supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"]
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="accounts_documents",
            query_name="match_accounts_documents"
        )
        
        # Setup agent
        self.agent_executor = self._create_agent()
    
    def search_with_filters(
        self,
        query: str,
        account_ids: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = 10
    ) -> str:
        """Search documents with filters."""
        try:
            search_kwargs = {"k": k}
            if account_ids:
                search_kwargs["account_ids"] = account_ids
            if document_types:
                search_kwargs["document_types"] = document_types
            if start_date:
                search_kwargs["start_date"] = start_date
            if end_date:
                search_kwargs["end_date"] = end_date
            
            retriever = FilteredSupabaseRetriever(
                vector_store=self.vector_store,
                search_kwargs=search_kwargs
            )
            
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant documents found."
            
            # Format response
            results = []
            for i, doc in enumerate(docs[:5], 1):
                results.append(
                    f"Result {i} (relevance: {doc.metadata.get('similarity', 0):.2%}):\n"
                    f"{doc.page_content[:400]}...\n"
                    f"Source: {doc.metadata.get('document_type')} | "
                    f"Account: {doc.metadata.get('account_id')}\n"
                )
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""
        tools = [
            Tool.from_function(
                func=self.search_with_filters,
                name="search_documents",
                description="Search company documents. Use filters for better results."
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that searches and analyzes company documents."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def query(self, question: str) -> str:
        """Query the agent."""
        result = self.agent_executor.invoke({"input": question})
        return result['output']

# Usage
if __name__ == "__main__":
    agent = DocumentSearchAgent()
    response = agent.query("What were the main topics discussed in TechCorp meetings last month?")
    print(response)
```

## Quick Reference: Direct RPC Call

If you need to call the function directly without LangChain wrappers:

```python
from supabase import create_client
from openai import OpenAI

# Initialize clients
supabase = create_client(supabase_url, supabase_key)
openai_client = OpenAI()

# Generate embedding
query = "What were the key decisions?"
embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Call match_accounts_documents
result = supabase.rpc(
    'match_accounts_documents',
    {
        'query_embedding': embedding,
        'match_count': 10,
        'filter': {
            'account_ids': ['acc_123', 'acc_456'],
            'document_types': ['meeting_notes']
        },
        'start_date': '2024-11-01T00:00:00Z',
        'end_date': None
    }
).execute()

# Process results
for doc in result.data:
    print(f"Similarity: {doc['similarity']:.3f}")
    print(f"Content: {doc['content'][:200]}...")
    print(f"Metadata: {doc['metadata']}")
    print("---")
```

This implementation provides a production-ready foundation for building AI agents that can intelligently search and reason over your vector store documents using LangChain.