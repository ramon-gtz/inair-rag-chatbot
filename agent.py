"""LangChain agent setup with OpenAI reasoning model."""
import os
import sys
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field as PydanticField
from supabase.client import create_client

# Force prints to go to stderr (terminal) instead of being captured by Streamlit
def debug_print(*args, **kwargs):
    """Print to stderr so it shows in terminal."""
    print(*args, file=sys.stderr, flush=True, **kwargs)
from config import (
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    OPENAI_MODEL, AGENT_MODEL, EMBEDDING_MODEL, TABLE_NAME, QUERY_NAME, DEFAULT_K,
    MAX_COMPREHENSIVE_K, STANDARD_K, REFINED_K
)
from retriever import FilteredSupabaseRetriever


class SearchInput(BaseModel):
    """Input schema for document search tool."""
    query: str = PydanticField(description="The search query")
    account_ids: Optional[List[str]] = PydanticField(
        default=None,
        description="List of account IDs to filter by (only use for technical IDs, not company names)"
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
    comprehensive: Optional[bool] = PydanticField(
        default=False,
        description="Set to True for comprehensive searches when user asks for 'all', 'every', 'complete', or wants thorough analysis. This retrieves many more results."
    )


class DocumentSearchAgent:
    """Agent for searching and reasoning over documents in the vector store."""
    
    def __init__(self, default_filters: Optional[Dict[str, Any]] = None):
        """Initialize the agent with Supabase and OpenAI clients.
        
        Args:
            default_filters: Optional dictionary of default filters to apply to all searches
        """
        self.default_filters = default_filters or {}
        
        # Ensure OpenAI API key is set in environment for LangChain to pick up
        import os
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Initialize Supabase client
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Initialize embeddings (will use OPENAI_API_KEY from env)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        
        # Initialize Agent LLM (gpt-4o-mini) - for tool calling and orchestration
        self.agent_llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=0
        )
        
        # Initialize Reasoning LLM (o3-mini) - for final analysis and reasoning
        # Note: o3-mini doesn't support temperature parameter
        self.reasoning_llm = ChatOpenAI(
            model=OPENAI_MODEL
        )
        
        # Keep self.llm for backward compatibility (use agent_llm)
        self.llm = self.agent_llm
        
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name=TABLE_NAME,
            query_name=QUERY_NAME
        )
        
        # Setup agent
        self.agent_executor = self._create_agent()
    
    def _multi_stage_search(self, strategy: Dict[str, Any], question: str) -> List[Document]:
        """Perform multi-stage search with different queries and filters."""
        all_docs = []
        seen_ids = set()
        
        # Determine k value based on comprehensive flag
        # Use smaller k for comprehensive to avoid timeouts, we'll do multiple searches instead
        # For comprehensive queries, use smaller k but do more search stages
        if strategy.get('comprehensive', False):
            k_value = MAX_COMPREHENSIVE_K  # Use 10 for comprehensive
        else:
            k_value = STANDARD_K  # Use 10 for standard
        
        # Build base filters
        base_filters = {}
        if strategy.get('document_types'):
            # Normalize document type names
            doc_types = strategy['document_types']
            # Map common variations to standard types
            normalized_types = []
            for dt in doc_types:
                dt_lower = dt.lower()
                if 'transcript' in dt_lower:
                    normalized_types.append('transcript')
                elif 'meeting' in dt_lower and 'note' in dt_lower:
                    normalized_types.append('meeting_notes')
                elif 'meeting' in dt_lower:
                    normalized_types.append('meeting_notes')
                elif 'email' in dt_lower:
                    normalized_types.append('email')
                elif 'message' in dt_lower:
                    normalized_types.append('message')
                else:
                    normalized_types.append(dt)
            base_filters['document_types'] = list(set(normalized_types))  # Remove duplicates
            debug_print(f"[DEBUG] Normalized document types: {strategy['document_types']} -> {base_filters['document_types']}")
        
        # Stage 1: Broad search with original question
        debug_print(f"[DEBUG] Stage 1: Broad search with k={k_value}")
        search_kwargs = {"k": k_value, **base_filters}
        retriever = FilteredSupabaseRetriever(
            vector_store=self.vector_store,
            search_kwargs=search_kwargs
        )
        docs_stage1 = retriever._get_relevant_documents(question)
        for doc in docs_stage1:
            doc_id = doc.metadata.get('id')
            if doc_id and doc_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc_id)
        debug_print(f"[DEBUG] Stage 1 retrieved {len(docs_stage1)} documents, {len(all_docs)} unique")
        
        # Stage 2: Search with query variations
        search_queries = strategy.get('search_queries', [question])
        if len(search_queries) > 1:
            for i, query_variant in enumerate(search_queries[1:], 2):
                debug_print(f"[DEBUG] Stage {i}: Search with variant '{query_variant[:50]}...'")
                retriever = FilteredSupabaseRetriever(
                    vector_store=self.vector_store,
                    search_kwargs=search_kwargs
                )
                docs_variant = retriever._get_relevant_documents(query_variant)
                for doc in docs_variant:
                    doc_id = doc.metadata.get('id')
                    if doc_id and doc_id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc_id)
                debug_print(f"[DEBUG] Stage {i} retrieved {len(docs_variant)} documents, {len(all_docs)} total unique")
        
        # Stage 3: Refined search with specific filters if client name identified
        if strategy.get('client_name') and strategy.get('document_types'):
            debug_print(f"[DEBUG] Stage 3: Refined search for {strategy['client_name']}")
            refined_query = f"{strategy['client_name']} {strategy['document_types'][0]}"
            refined_kwargs = {"k": REFINED_K, **base_filters}
            retriever = FilteredSupabaseRetriever(
                vector_store=self.vector_store,
                search_kwargs=refined_kwargs
            )
            docs_refined = retriever._get_relevant_documents(refined_query)
            for doc in docs_refined:
                doc_id = doc.metadata.get('id')
                if doc_id and doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)
            debug_print(f"[DEBUG] Stage 3 retrieved {len(docs_refined)} documents, {len(all_docs)} total unique")
        
        debug_print(f"[MULTI-STAGE SEARCH] Complete: {len(all_docs)} unique documents retrieved")
        return all_docs
    
    def _group_documents_by_meeting(self, docs: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by meeting_id, document_id, or date."""
        from datetime import datetime
        
        grouped = {}
        
        for doc in docs:
            metadata = doc.metadata
            doc_type = metadata.get('document_type', '').lower()
            
            # Try to find grouping key in order of preference
            group_key = None
            
            # For transcripts and meeting documents, treat each document as its own meeting
            # unless there's an explicit meeting_id
            if 'transcript' in doc_type or 'meeting' in doc_type:
                # Priority 1: meeting_id (if multiple transcripts share a meeting)
                if metadata.get('meeting_id'):
                    group_key = f"meeting_{metadata['meeting_id']}"
                # Priority 2: document_id (each transcript is typically its own meeting)
                elif metadata.get('document_id'):
                    group_key = f"transcript_{metadata['document_id']}"
                # Priority 3: Use document ID as fallback
                elif metadata.get('id'):
                    group_key = f"transcript_{metadata['id']}"
                else:
                    # Last resort: use content hash or index
                    doc_id = f"transcript_{len(grouped)}"
                    group_key = doc_id
            else:
                # For non-transcript documents, use standard grouping
                # Priority 1: meeting_id
                if metadata.get('meeting_id'):
                    group_key = f"meeting_{metadata['meeting_id']}"
                # Priority 2: document_id
                elif metadata.get('document_id'):
                    group_key = f"doc_{metadata['document_id']}"
                # Priority 3: date-based grouping (group by day)
                elif metadata.get('created_at') or metadata.get('date'):
                    date_str = metadata.get('created_at') or metadata.get('date')
                    try:
                        # Parse date and group by day
                        if isinstance(date_str, str):
                            # Try ISO format
                            if 'T' in date_str:
                                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            else:
                                date_obj = datetime.fromisoformat(date_str)
                            group_key = f"date_{date_obj.strftime('%Y-%m-%d')}"
                        else:
                            group_key = "date_unknown"
                    except:
                        group_key = "date_unknown"
                # Priority 4: account_id + date combination
                elif metadata.get('account_id'):
                    account_id = metadata['account_id']
                    date_str = metadata.get('created_at') or metadata.get('date', 'unknown')
                    group_key = f"account_{account_id}_{date_str}"
                # Fallback: use document ID
                else:
                    doc_id = metadata.get('id', f"doc_{len(grouped)}")
                    group_key = f"fallback_{doc_id}"
            
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(doc)
        
        # Sort documents within each group by date if available
        for group_key, group_docs in grouped.items():
            group_docs.sort(key=lambda d: (
                d.metadata.get('created_at') or d.metadata.get('date') or ''
            ), reverse=True)
        
        debug_print(f"[GROUPING] Grouped {len(docs)} documents into {len(grouped)} groups")
        for group_id, group_docs in list(grouped.items())[:5]:  # Show first 5 groups
            debug_print(f"  - {group_id}: {len(group_docs)} docs, type: {group_docs[0].metadata.get('document_type', 'unknown')}")
        return grouped
    
    def _analyze_meeting_group(self, group_id: str, docs: List[Document], question: str) -> Dict[str, Any]:
        """Analyze a group of documents (meeting) using o3-mini to summarize what was done for the client."""
        # For transcripts, include full content (they're typically single documents per meeting)
        # For other types, limit to avoid token issues
        first_doc = docs[0]
        doc_type = first_doc.metadata.get('document_type', '').lower()
        
        if 'transcript' in doc_type and len(docs) == 1:
            # Single transcript - include full content
            combined_content = docs[0].page_content
        else:
            # Multiple documents or non-transcript - combine with limits
            max_docs = 10 if 'transcript' not in doc_type else len(docs)
            max_chars_per_doc = 10000 if 'transcript' in doc_type else 5000
            combined_parts = []
            for i, doc in enumerate(docs[:max_docs]):
                content = doc.page_content[:max_chars_per_doc]
                combined_parts.append(f"Document {i+1}:\n{content}")
            combined_content = "\n\n---\n\n".join(combined_parts)
        
        # Get metadata from first document
        first_doc = docs[0]
        metadata = first_doc.metadata
        
        # Create analysis prompt for o3-mini
        # Determine content limit based on document type
        content_limit = 50000 if 'transcript' in doc_type else 8000
        
        analysis_prompt = f"""You are analyzing a meeting or set of related documents from InAir Studio's client interactions.

Original question: {question}

Meeting/Document Group Information:
- Group ID: {group_id}
- Document Type: {metadata.get('document_type', 'unknown')}
- Account: {metadata.get('account_id', 'unknown')}
- Date: {metadata.get('created_at') or metadata.get('date', 'unknown')}
- Number of documents: {len(docs)}

Meeting Content:
{combined_content[:content_limit]}

Analyze this meeting/group and provide a structured summary answering:
1. What was done for the client in this meeting?
2. What were the key actions, decisions, or outcomes?
3. What topics were discussed?
4. What follow-up items or next steps were mentioned?

Provide a clear, concise summary focusing on what was accomplished for the client."""
        
        try:
            response = self.reasoning_llm.invoke([HumanMessage(content=analysis_prompt)])
            summary = response.content
            
            return {
                'group_id': group_id,
                'date': metadata.get('created_at') or metadata.get('date', 'unknown'),
                'document_type': metadata.get('document_type', 'unknown'),
                'account_id': metadata.get('account_id', 'unknown'),
                'num_documents': len(docs),
                'summary': summary,
                'metadata': metadata
            }
        except Exception as e:
            debug_print(f"[DEBUG] Error analyzing meeting group {group_id}: {str(e)}")
            return {
                'group_id': group_id,
                'date': metadata.get('created_at') or metadata.get('date', 'unknown'),
                'document_type': metadata.get('document_type', 'unknown'),
                'account_id': metadata.get('account_id', 'unknown'),
                'num_documents': len(docs),
                'summary': f"Error analyzing this meeting: {str(e)}",
                'metadata': metadata
            }
    
    def _synthesize_breakdown(self, question: str, meeting_summaries: List[Dict[str, Any]]) -> str:
        """Use o3-mini to create final per-meeting breakdown from summaries."""
        import json
        
        # Format summaries for the prompt
        summaries_text = "\n\n".join([
            f"Meeting {i+1} ({summary.get('date', 'unknown date')}):\n"
            f"Type: {summary.get('document_type', 'unknown')}\n"
            f"Account: {summary.get('account_id', 'unknown')}\n"
            f"Summary: {summary.get('summary', 'No summary available')}"
            for i, summary in enumerate(meeting_summaries)
        ])
        
        synthesis_prompt = f"""You are creating a comprehensive per-meeting breakdown for InAir Studio.

Original question: {question}

You have analyzed {len(meeting_summaries)} meetings/documents. Here are the summaries:

{summaries_text}

Create a clear, organized per-meeting breakdown that:
1. Lists each meeting separately
2. For each meeting, clearly states what was done for the client
3. Organizes information chronologically if dates are available
4. Highlights key actions, decisions, and outcomes for each meeting
5. Provides a comprehensive overview that answers the original question

Format the response as a well-structured breakdown with clear sections for each meeting."""
        
        try:
            response = self.reasoning_llm.invoke([HumanMessage(content=synthesis_prompt)])
            return response.content
        except Exception as e:
            debug_print(f"[DEBUG] Error synthesizing breakdown: {str(e)}")
            # Fallback: return formatted summaries
            breakdown = f"Per-Meeting Breakdown ({len(meeting_summaries)} meetings):\n\n"
            for i, summary in enumerate(meeting_summaries, 1):
                breakdown += f"Meeting {i} ({summary.get('date', 'unknown date')}):\n"
                breakdown += f"{summary.get('summary', 'No summary available')}\n\n"
            return breakdown
    
    def _combine_documents(self, docs: List[Document], max_chars: int = 50000) -> str:
        """Combine document contents into a single string, respecting token limits."""
        combined = []
        total_chars = 0
        
        for i, doc in enumerate(docs):
            doc_text = f"\n\n--- Document {i+1} ---\n"
            doc_text += f"Type: {doc.metadata.get('document_type', 'unknown')}\n"
            doc_text += f"Account: {doc.metadata.get('account_id', 'unknown')}\n"
            doc_text += f"Date: {doc.metadata.get('created_at') or doc.metadata.get('date', 'unknown')}\n"
            doc_text += f"Content: {doc.page_content}\n"
            
            if total_chars + len(doc_text) > max_chars:
                break
            combined.append(doc_text)
            total_chars += len(doc_text)
        
        return "".join(combined)
    
    def _handle_comprehensive_query(self, question: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive queries with multi-stage search and breakdown."""
        debug_print(f"\n[COMPREHENSIVE QUERY] Strategy: {strategy}")
        
        # Stage 1: Multi-stage retrieval
        all_docs = self._multi_stage_search(strategy, question)
        
        if not all_docs:
            return {
                'output': "No relevant documents found for this comprehensive query.",
                'sources': [],
                'total_documents': 0,
                'total_groups': 0
            }
        
        # Filter documents by document_type if specified in strategy
        filtered_docs = all_docs
        if strategy.get('document_types'):
            doc_types = [dt.lower().strip() for dt in strategy['document_types']]
            # Also check for variations like "meeting transcript" matching "transcript"
            filtered_docs = []
            for doc in all_docs:
                doc_type = doc.metadata.get('document_type', '').lower().strip()
                # Check if any of the requested types match (including partial matches)
                matches = False
                for req_type in doc_types:
                    # Exact match or substring match (e.g., "transcript" matches "meeting transcript")
                    if req_type == doc_type or req_type in doc_type or doc_type in req_type:
                        matches = True
                        break
                if matches:
                    filtered_docs.append(doc)
            debug_print(f"\n[FILTERING] Filtered {len(all_docs)} documents to {len(filtered_docs)} matching document types: {strategy['document_types']}")
            if filtered_docs:
                debug_print(f"[FILTERING] Sample filtered doc types: {[doc.metadata.get('document_type') for doc in filtered_docs[:10]]}")
                debug_print(f"[FILTERING] Unique doc types in filtered: {set([doc.metadata.get('document_type') for doc in filtered_docs])}")
                debug_print(f"[FILTERING] Total transcripts found: {len([d for d in filtered_docs if 'transcript' in d.metadata.get('document_type', '').lower()])}")
            else:
                debug_print(f"[FILTERING] WARNING: No documents matched document types!")
                debug_print(f"[FILTERING] Available types in all_docs: {set([doc.metadata.get('document_type') for doc in all_docs[:20]])}")
        
        if not filtered_docs:
            return {
                'output': f"No documents found matching the requested document types: {strategy.get('document_types', [])}",
                'sources': [],
                'total_documents': len(all_docs),
                'total_groups': 0
            }
        
        # Stage 2: Group documents if breakdown required
        if strategy.get('requires_breakdown', False):
            grouped = self._group_documents_by_meeting(filtered_docs)
            
            # Stage 3: Analyze each group with o3-mini
            debug_print(f"\n[ANALYSIS] Analyzing {len(grouped)} groups...")
            summaries = []
            for group_id, docs in grouped.items():
                summary = self._analyze_meeting_group(group_id, docs, question)
                summaries.append(summary)
            
            # Stage 4: Synthesize final breakdown with o3-mini
            debug_print(f"[SYNTHESIS] Creating final breakdown from {len(summaries)} meeting summaries...")
            final_answer = self._synthesize_breakdown(question, summaries)
            
            return {
                'output': final_answer,
                'sources': summaries,
                'total_documents': len(all_docs),
                'total_groups': len(grouped)
            }
        else:
            # Comprehensive but no breakdown needed - use o3-mini to analyze all documents
            debug_print(f"[DEBUG] Comprehensive query without breakdown - analyzing {len(all_docs)} documents...")
            combined_content = self._combine_documents(all_docs)
            
            analysis_prompt = f"""Analyze these documents and answer the question comprehensively.

Question: {question}

Documents ({len(all_docs)} total):
{combined_content}

Provide a thorough, comprehensive answer based on all the documents provided."""
            
            answer = self.reasoning_llm.invoke([HumanMessage(content=analysis_prompt)])
            
            return {
                'output': answer.content,
                'sources': all_docs,
                'total_documents': len(all_docs),
                'total_groups': 0
            }
    
    def search_with_filters(
        self,
        query: str,
        account_ids: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        comprehensive: Optional[bool] = False
    ) -> str:
        """Search documents in the vector store with optional filters."""
        debug_print(f"[DEBUG] search_with_filters called with query: {query[:50]}...")
        debug_print(f"[DEBUG] Filters - account_ids: {account_ids}, document_types: {document_types}, comprehensive: {comprehensive}")
        try:
            # Build search kwargs, starting with defaults
            # Use much higher k for comprehensive searches (e.g., "all meetings", "every transcript")
            if comprehensive:
                k_value = 100  # Get many more results for thorough analysis
                debug_print(f"[DEBUG] Comprehensive search mode - retrieving up to {k_value} documents")
            else:
                k_value = 20  # Standard search
            search_kwargs = {"k": k_value}
            
            # Apply default filters first (only if they have values)
            default_account_ids = self.default_filters.get('account_ids')
            if default_account_ids and len(default_account_ids) > 0:
                search_kwargs["account_ids"] = default_account_ids
            
            default_document_types = self.default_filters.get('document_types')
            if default_document_types and len(default_document_types) > 0:
                search_kwargs["document_types"] = default_document_types
            
            if self.default_filters.get('start_date'):
                search_kwargs["start_date"] = self.default_filters['start_date']
            if self.default_filters.get('end_date'):
                search_kwargs["end_date"] = self.default_filters['end_date']
            
            # Override with explicit parameters if provided
            if account_ids and len(account_ids) > 0:
                search_kwargs["account_ids"] = account_ids
            if document_types and len(document_types) > 0:
                search_kwargs["document_types"] = document_types
            if start_date:
                search_kwargs["start_date"] = start_date
            if end_date:
                search_kwargs["end_date"] = end_date
            
            # Create retriever
            retriever = FilteredSupabaseRetriever(
                vector_store=self.vector_store,
                search_kwargs=search_kwargs
            )
            
            # Get documents (BaseRetriever.get_relevant_documents calls _get_relevant_documents)
            docs = retriever._get_relevant_documents(query)
            
            # If no results, try a broader search without filters (if filters were applied)
            if not docs and (search_kwargs.get("account_ids") or search_kwargs.get("document_types")):
                debug_print(f"[DEBUG] No results with filters, trying broader search...")
                broad_k = 100 if comprehensive else 20
                broad_search_kwargs = {"k": broad_k}
                if start_date:
                    broad_search_kwargs["start_date"] = start_date
                if end_date:
                    broad_search_kwargs["end_date"] = end_date
                
                broad_retriever = FilteredSupabaseRetriever(
                    vector_store=self.vector_store,
                    search_kwargs=broad_search_kwargs
                )
                docs = broad_retriever._get_relevant_documents(query)
            
            # Format results - show more documents and lower similarity threshold
            if not docs:
                return "No relevant documents found. The search query may not match any documents in the database, or the client name might be stored differently."
            
            results = []
            # For comprehensive searches, show all results; for standard searches, show top 10
            max_docs_to_show = len(docs) if comprehensive else min(10, len(docs))
            
            for i, doc in enumerate(docs[:max_docs_to_show], 1):
                similarity = doc.metadata.get('similarity', 0)
                doc_type = doc.metadata.get('document_type', 'unknown')
                account_id = doc.metadata.get('account_id', 'unknown')
                doc_id = doc.metadata.get('id', 'unknown')
                
                # For comprehensive searches, show more content; for standard, show preview
                if comprehensive:
                    content_preview = doc.page_content[:1500]  # Much longer for comprehensive
                else:
                    content_preview = doc.page_content[:500]
                
                # Include document even if similarity is lower - let the AI decide relevance
                results.append(
                    f"Document {i} (ID: {doc_id}, similarity: {similarity:.3f}):\n"
                    f"Type: {doc_type}\n"
                    f"Account: {account_id}\n"
                    f"Content: {content_preview}{'...' if len(doc.page_content) > len(content_preview) else ''}\n"
                )
            
            result_text = "\n---\n".join(results)
            
            # Add summary for comprehensive searches
            if comprehensive:
                result_text += f"\n\n[Total documents retrieved: {len(docs)}]"
            
            return result_text
            
        except Exception as e:
            debug_print(f"[DEBUG] Search error: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def _analyze_query_strategy(self, question: str) -> Dict[str, Any]:
        """Analyze query using gpt-4o-mini to determine search strategy."""
        import json
        import re
        
        analysis_prompt = f"""Analyze this question and determine the search strategy. Return ONLY valid JSON, no other text.

Question: {question}

Analyze and return JSON with these fields:
- "comprehensive": boolean - true if question asks for "all", "every", "complete", "entire", or wants thorough analysis
- "document_types": array of strings - document types mentioned (e.g., ["transcript", "meeting_notes", "email"]) or empty array if not specified
- "client_name": string or null - client/project name mentioned (e.g., "Fox", "Rolex", "Nike") or null if not mentioned
- "requires_breakdown": boolean - true if question asks for "per meeting", "per-document", "breakdown", "each meeting", etc.
- "search_queries": array of strings - 2-3 search query variations to use (e.g., ["Fox project transcripts", "Fox meetings", "Fox project discussions"])

Example response format:
{{"comprehensive": true, "document_types": ["transcript"], "client_name": "Fox", "requires_breakdown": true, "search_queries": ["Fox project transcripts", "Fox meetings", "Fox project discussions"]}}

Return JSON:"""
        
        try:
            response = self.agent_llm.invoke([HumanMessage(content=analysis_prompt)])
            content = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
            
            strategy = json.loads(json_str)
            
            # Validate and set defaults
            strategy.setdefault('comprehensive', False)
            strategy.setdefault('document_types', [])
            strategy.setdefault('client_name', None)
            strategy.setdefault('requires_breakdown', False)
            strategy.setdefault('search_queries', [question])  # Default to original question
            
            debug_print(f"[DEBUG] Query strategy: {strategy}")
            return strategy
            
        except Exception as e:
            debug_print(f"[DEBUG] Error analyzing query strategy: {str(e)}")
            # Fallback strategy
            question_lower = question.lower()
            return {
                'comprehensive': any(word in question_lower for word in ['all', 'every', 'complete', 'entire']),
                'document_types': [],
                'client_name': None,
                'requires_breakdown': any(phrase in question_lower for phrase in ['per meeting', 'per-meeting', 'breakdown', 'each meeting']),
                'search_queries': [question]
            }
    
    def _create_agent(self):
        """Create the LangChain agent with document search tool."""
        # Create search tool
        search_tool = StructuredTool.from_function(
            func=self.search_with_filters,
            name="search_documents",
            description="""MANDATORY: You MUST use this tool to search for information before answering any question.

Search through InAir Studio's client documents and conversations using semantic search.

CRITICAL DISTINCTION:
- Account IDs are technical identifiers like "acc_123", "acc_456" - users don't know these
- Account Names are company names like "Rolex", "Nike", "Apple", "Fox" - these appear in DOCUMENT CONTENT
- When a user mentions a company/project name (e.g., "Rolex", "Fox project"), DO NOT use account_ids filter
- Company/project names are found in document content via semantic search, not in account_id fields
- Only use account_ids filter if the user explicitly provides a technical account ID

COMPREHENSIVE SEARCH MODE:
- Set comprehensive=True when user asks for:
  * "ALL" documents/meetings/transcripts
  * "EVERY" meeting/email/conversation
  * "COMPLETE" analysis or breakdown
  * Per-meeting/per-document breakdowns
  * Thorough research or exhaustive search
- Comprehensive mode retrieves up to 100 documents instead of 20
- Use comprehensive mode for queries like:
  * "Go across ALL meeting transcripts within the Fox project"
  * "Give me a per meeting breakdown"
  * "Show me every conversation with [client]"
  * "Analyze all documents related to [project]"

Use this tool to find:
- Client conversations and interactions (company names are in the content)
- Meeting notes and transcripts  
- Email communications
- Project details and updates
- Any client-related information

FILTERING (all optional - leave empty to search broadly):
- account_ids: ONLY use if user provides exact technical account ID (e.g., ["acc_123"]) - NOT for company names
- document_types: Use for specific types like ["transcript", "meeting_notes", "email"] when user specifies
- start_date/end_date: ISO format dates for time filtering
- comprehensive: Set to True for thorough "all/every/complete" queries

SEARCH STRATEGY:
- For company/project name queries (e.g., "Fox project", "Rolex", "What did we discuss with Nike?"):
  * Search WITHOUT account_ids filter
  * Use the company/project name in the query - semantic search finds it in document content
  * If user asks for "ALL" or wants thorough analysis, set comprehensive=True
- For queries asking for specific document types (e.g., "all meeting transcripts"), use document_types filter
- For technical account ID queries (e.g., "acc_123"), then use account_ids filter
- When user wants per-meeting/per-document breakdowns, use comprehensive=True and appropriate document_types
- Always search - never skip this step""",
            args_schema=SearchInput
        )
        
        tools = [search_tool]
        
        # System prompt for the agent
        system_prompt = """You are an AI assistant for InAir Studio, helping agents access and analyze client conversation data and related documents.

CRITICAL: You MUST ALWAYS use the search_documents tool to answer questions. Never say you don't have information without searching first.

IMPORTANT DISTINCTION:
- Account IDs are technical identifiers (e.g., "acc_123", "acc_456") - DO NOT use these when a company name is mentioned
- Account Names are company/client names (e.g., "Rolex", "Nike", "Apple") - these appear in DOCUMENT CONTENT, not in account IDs
- When a user mentions a company name like "Rolex", search for it in the document content using semantic search
- DO NOT try to filter by account_ids when a company name is mentioned - account names are in the content, not in IDs

CONTEXT:
- You work for InAir Studio, a company that manages client relationships and conversations
- The database contains conversations, meeting notes, emails, transcripts, and other client-related documents
- Documents are organized by accounts (clients) and document types
- Client/company names (like "Rolex", "Nike", etc.) appear in the DOCUMENT CONTENT, not in account_id fields
- Account IDs are technical identifiers that users don't typically know

WORKFLOW:
1. When a user asks ANY question, you MUST first call search_documents tool
2. Use the user's question as the search query - semantic search will find company names in document content
3. If a company name is mentioned (e.g., "Rolex"), search for it WITHOUT using account_ids filter
4. The semantic search will find mentions of the company name in document content automatically
5. Review the search results
6. Provide an answer based on the retrieved documents

SEARCH STRATEGY:
- ALWAYS search first - never respond without searching
- When a COMPANY/PROJECT NAME is mentioned (e.g., "Rolex", "Nike", "Fox project"), search WITHOUT account_ids filter - names are in content
- When user asks for "ALL", "EVERY", "COMPLETE", or wants thorough analysis, use comprehensive=True
- When user wants per-meeting/per-document breakdowns, use comprehensive=True and filter by document_types
- Use account_ids filter ONLY if the user provides a technical account ID (like "acc_123")
- Use document_types filter when user specifically asks for certain types (meetings, emails, transcripts, etc.)
- Use date filters when time context is mentioned (recent, last week, etc.)
- Semantic search will find company/project names in document content automatically
- For complex queries requiring multiple documents (e.g., "all meetings", "every transcript"), use comprehensive mode

RESPONSE STYLE:
- Always search before answering
- Provide clear, concise answers based on retrieved documents
- Cite specific documents when referencing information
- If search returns no results, try a different search query or broader search
- Be professional and helpful, as you're representing InAir Studio"""
        
        # Create agent using the new API
        # Use agent_llm (gpt-4o-mini) for tool calling and orchestration
        # o3-mini (reasoning_llm) will be used for final analysis in comprehensive queries
        agent = create_agent(
            self.agent_llm,  # Use gpt-4o-mini for tool calling
            tools,
            system_prompt=system_prompt
        )
        
        return agent
    
    def query(self, question: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Query the agent and return response with sources.
        
        Args:
            question: The user's question
            chat_history: Optional list of previous messages to include in context
        
        Returns:
            Dictionary with 'output' (response text) and 'sources' (list of documents or summaries)
        """
        try:
            # Step 1: Analyze query to determine strategy
            debug_print(f"\n{'='*80}")
            debug_print(f"[QUERY START] Analyzing query: {question}")
            debug_print(f"{'='*80}")
            strategy = self._analyze_query_strategy(question)
            
            # Step 2: Route to appropriate handler
            if strategy.get('comprehensive', False):
                # Use comprehensive handler with multi-stage search
                debug_print(f"[DEBUG] Routing to comprehensive query handler")
                return self._handle_comprehensive_query(question, strategy)
            else:
                # Use standard agent-based approach
                debug_print(f"[DEBUG] Routing to standard agent handler")
                return self._handle_standard_query(question, chat_history, strategy)
            
        except Exception as e:
            debug_print(f"[DEBUG] Error in query method: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'output': f"Error processing query: {str(e)}",
                'sources': []
            }
    
    def _handle_standard_query(self, question: str, chat_history: Optional[List[Dict]], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard queries using the agent pattern."""
        # Build messages list with chat history
        messages = []
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
        
        # Add current question
        messages.append(HumanMessage(content=question))
        
        # Invoke agent with messages
        debug_print(f"[DEBUG] Invoking agent with question: {question[:50]}...")
        result = self.agent_executor.invoke({"messages": messages})
        debug_print(f"[DEBUG] Agent result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Extract output from result
        # The result is a state dict with 'messages' list
        output_messages = result.get('messages', [])
        debug_print(f"[DEBUG] Number of messages in result: {len(output_messages)}")
        # Get the last AI message
        output = ""
        sources = []
        
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage):
                output = msg.content
                break
        
        # Extract sources from tool messages and intermediate steps
        for msg in output_messages:
            # Check for tool messages
            if hasattr(msg, 'name') and msg.name == 'search_documents':
                if hasattr(msg, 'content'):
                    sources.append(str(msg.content))
            # Also check content for document references
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                if 'Document' in msg.content or 'similarity' in msg.content.lower():
                    sources.append(msg.content)
        
        return {
            'output': output,
            'sources': sources
        }
    
    def clear_memory(self):
        """Clear the conversation memory (no-op, handled via chat_history parameter)."""
        pass
    
    def update_default_filters(self, filters: Dict[str, Any]):
        """Update the default filters applied to all searches.
        
        Args:
            filters: Dictionary with keys: account_ids, document_types, start_date, end_date
        """
        self.default_filters = filters

