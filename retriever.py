"""Custom retriever for Supabase vector store with filtering support."""
import sys
import time
from typing import List, Optional, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from pydantic import Field
from config import MIN_SIMILARITY_THRESHOLD, RPC_TIMEOUT_SECONDS, PROGRESSIVE_K_VALUES

# Force prints to go to stderr (terminal)
def debug_print(*args, **kwargs):
    """Print to stderr so it shows in terminal."""
    print(*args, file=sys.stderr, flush=True, **kwargs)


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
        
        if "document_type" in self.search_kwargs:
            filter_obj["document_type"] = self.search_kwargs["document_type"]
        
        if "document_id" in self.search_kwargs:
            filter_obj["document_id"] = self.search_kwargs["document_id"]
        
        # Get k (number of results)
        k = self.search_kwargs.get("k", 10)
        
        # Generate embedding for the query
        # Access embedding from the vector store's embedding attribute
        # SupabaseVectorStore stores the embedding model in _embedding or we need to pass it
        if hasattr(self.vector_store, 'embeddings'):
            embedding_model = self.vector_store.embeddings
        elif hasattr(self.vector_store, '_embeddings'):
            embedding_model = self.vector_store._embeddings
        else:
            # Try to get from the embedding attribute directly
            embedding_model = getattr(self.vector_store, 'embedding', None)
        
        if embedding_model is None:
            raise ValueError("Could not access embedding model from vector store")
        
        query_embedding = embedding_model.embed_query(query)
        
        # Access the Supabase client from the vector store
        supabase_client = getattr(self.vector_store, 'client', None) or getattr(self.vector_store, '_client', None)
        if supabase_client is None:
            raise ValueError("Could not access Supabase client from vector store")
        
        # Progressive retry strategy: try smaller k values if timeout occurs
        # Start with requested k, but if timeout, progressively reduce
        k_values_to_try = [k] + [prog_k for prog_k in PROGRESSIVE_K_VALUES if prog_k < k]
        
        for attempt, current_k in enumerate(k_values_to_try):
            try:
                debug_print(f"[DEBUG] Attempt {attempt + 1}: Calling Supabase RPC with k={current_k}")
                debug_print(f"[DEBUG] Query: {query[:50]}..., filters: {filter_obj if filter_obj else 'None'}")
                
                # Record start time for timeout detection
                start_time = time.time()
                
                # Make RPC call
                results = supabase_client.rpc(
                    'match_accounts_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_count': current_k,
                        'filter': filter_obj if filter_obj else None,
                        'start_date': self.search_kwargs.get('start_date'),
                        'end_date': self.search_kwargs.get('end_date')
                    }
                ).execute()
                
                elapsed = time.time() - start_time
                debug_print(f"[DEBUG] RPC call completed in {elapsed:.2f}s, returned {len(results.data) if results.data else 0} documents")
                
                # Convert to LangChain documents and filter by similarity threshold
                documents = []
                if results.data:
                    for result in results.data:
                        similarity = result.get('similarity', 0.0)
                        # Filter by minimum similarity threshold
                        if similarity >= MIN_SIMILARITY_THRESHOLD:
                            doc = Document(
                                page_content=result.get('content', ''),
                                metadata={
                                    **(result.get('metadata', {}) or {}),
                                    'similarity': similarity,
                                    'id': result.get('id')
                                }
                            )
                            documents.append(doc)
                        else:
                            debug_print(f"[DEBUG] Filtered out document with similarity {similarity:.3f} < {MIN_SIMILARITY_THRESHOLD}")
                
                debug_print(f"[DEBUG] After similarity filtering: {len(documents)} documents (threshold: {MIN_SIMILARITY_THRESHOLD})")
                return documents
                
            except Exception as e:
                error_msg = str(e)
                elapsed = time.time() - start_time if 'start_time' in locals() else RPC_TIMEOUT_SECONDS
                
                # Check if it's a timeout error
                is_timeout = (
                    'timeout' in error_msg.lower() or 
                    '57014' in error_msg or
                    elapsed >= RPC_TIMEOUT_SECONDS
                )
                
                if is_timeout:
                    debug_print(f"[DEBUG] Timeout detected (elapsed: {elapsed:.2f}s): {error_msg[:100]}")
                    # If we have more k values to try, continue to next iteration
                    if attempt < len(k_values_to_try) - 1:
                        next_k = k_values_to_try[attempt + 1]
                        debug_print(f"[DEBUG] Retrying with smaller k={next_k}...")
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    else:
                        debug_print(f"[DEBUG] All retry attempts failed")
                        return []
                else:
                    # Non-timeout error - log and return empty
                    debug_print(f"[DEBUG] Non-timeout error: {error_msg[:200]}")
                    return []
        
        # Should never reach here, but return empty if we do
        return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of retrieval."""
        return self._get_relevant_documents(query)

