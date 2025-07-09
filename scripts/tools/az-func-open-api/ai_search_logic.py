import azure.functions as func
import logging
import os
import requests
import json
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from typing import Optional
from models.models import user_query_request, document, search_response

# Global variables to store clients (initialized once per function instance)
_search_client: Optional[SearchClient] = None
_embeddings_model: Optional[AzureOpenAIEmbeddings] = None

NUM_SEARCH_RESULTS = 15
K_NEAREST_NEIGHBORS = 30

def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """Get or create the Azure OpenAI embeddings model (singleton pattern)"""
    global _embeddings_model
    if _embeddings_model is None:
        logging.info("Initializing Azure OpenAI embeddings model...")
        
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = os.environ.get("AZURE_OPENAI_KEY")
        model_name = os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")
        
        logging.info(f"Using endpoint: {endpoint}")
        logging.info(f"Using model: {model_name}")
        # logging.info(f"Key configured: {'Yes' if key else 'No'}")
        logging.info(f"Key configured: {key}")
        
        if not all([endpoint, key, model_name]):
            missing = []
            if not endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
            if not key: missing.append("AZURE_OPENAI_KEY")
            if not model_name: missing.append("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
            
            error_msg = f"Missing required Azure OpenAI configuration: {', '.join(missing)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            _embeddings_model = AzureOpenAIEmbeddings(
                azure_endpoint=endpoint,
                api_key=key,
                azure_deployment=model_name
            )
            logging.info("✅ Azure OpenAI embeddings model initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Azure OpenAI embeddings model: {str(e)}")
            raise
    
    return _embeddings_model
def get_search_client() -> SearchClient:
    """Get or create the Azure AI Search client (singleton pattern)"""
    global _search_client
    if _search_client is None:
        logging.info("Initializing Azure AI Search client...")
        
        endpoint = os.environ.get("AZURE_AI_SEARCH_ENDPOINT")
        key = os.environ.get("AZURE_AI_SEARCH_KEY")
        index_name = os.environ.get("AZURE_AI_SEARCH_INDEX_NAME")
        
        logging.info(f"Using search endpoint: {endpoint}")
        logging.info(f"Using index: {index_name}")
        logging.info(f"Key configured: {'Yes' if key else 'No'}")
        
        if not all([endpoint, key, index_name]):
            missing = []
            if not endpoint: missing.append("AZURE_AI_SEARCH_ENDPOINT")
            if not key: missing.append("AZURE_AI_SEARCH_KEY")
            if not index_name: missing.append("AZURE_AI_SEARCH_INDEX_NAME")
            
            error_msg = f"Missing required Azure AI Search configuration: {', '.join(missing)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            credential = AzureKeyCredential(key)
            _search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=credential
            )
            logging.info("✅ Azure AI Search client initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Azure AI Search client: {str(e)}")
            raise
    
    return _search_client

def execute_ai_search(request_id: str, user_query: str) -> dict:
    # Get the initialized clients
    search_client = get_search_client()
    logging.info(f'[{request_id}] ✅ Search client initialized successfully')
        
    embeddings_model = get_embeddings_model()
    logging.info(f'[{request_id}] ✅ Embeddings model initialized successfully')

    logging.info(f'[{request_id}] Step 2: Parsing request body...')
    
    logging.info(f'[{request_id}] Step 3a: Generating vector embedding...')
                
    # Generate vector embedding for the query
    query_vector = embeddings_model.embed_query(user_query)
    vector_length = len(query_vector) if query_vector else 0
    logging.info(f'[{request_id}] ✅ Vector embedding generated successfully (dimension: {vector_length})')
    
    logging.info(f'[{request_id}] Step 3b: Creating vector query...')
    
    # Create vector queries for all three vector fields
    vector_queries = [
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=K_NEAREST_NEIGHBORS,
            fields="content_vector"
        )
    ]
    logging.info(f'[{request_id}] ✅ Vector query created (k={K_NEAREST_NEIGHBORS}, fields=content_vector)')
    
    logging.info(f'[{request_id}] Step 3c: Executing search against Azure AI Search...')
    
    # Perform the search with all vector fields and corresponding text fields
    results = search_client.search(
        search_text=user_query,
        vector_queries=vector_queries,
        select=["id", "name", "architecture_url", "content"],
        top=NUM_SEARCH_RESULTS
    )
    logging.info(f'[{request_id}] ✅ Search executed successfully, processing results...')
    
    logging.info(f'[{request_id}] Step 3d: Processing search results...')
    
    search_results: list = []
    result_count = 0
    
    for result in results:
        result_count += 1
        logging.debug(f'[{request_id}] Processing result {result_count}: ID={result.get("id", "unknown")}')
        
        # Combine all text content for the LLM with clear delineation
        content_parts = []
        
        # Always include title at the top
        if result.get("name"):
            content_parts.append(f"=== NAME ===\n{result['name']}\n=== END NAME ===")
        
        if result.get("architecture_url"):
            content_parts.append(f"=== URL ===\n{result['architecture_url']}\n=== END URL ===")

        if result.get("content"):
            content_parts.append(f"=== CONTENT ===\n{result['content']}\n=== END CONTENT ===")
        
        combined_content = "\n\n".join(content_parts)
        logging.info(f'[{request_id}] Combined content for result {result_count}: {combined_content}')
        # search_result = {
        #     "id": result["id"],
        #     "name": result['name'],
        #     "architecture_url": result.get("architecture_url", ""),
        #     "content": combined_content,
        #     "score": result["@search.score"]
        # }
        # Create a document dictionary (JSON serializable)
        search_result = {
            "id": result["id"],
            "name": result["name"],
            "architecture_url": result.get("architecture_url", ""),
            "content": combined_content,
            "score": result["@search.score"]
        }
        search_results.append(search_result)

    logging.info(f'[{request_id}] ✅ Successfully processed {len(search_results)} search results')
    logging.info(f'[{request_id}] Step 4: Returning response with {len(search_results)} results')

    return {
        "documents": search_results
    }
