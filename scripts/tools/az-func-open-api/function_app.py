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
from ai_search_logic import execute_ai_search
from models.models import user_query_request, status_get_uri_response, search_response, status_get_uri_request
from azure_functions_openapi.decorator import openapi
from azure_functions_openapi.openapi import get_openapi_json
from azure_functions_openapi.swagger_ui import render_swagger_ui
from durable_blueprints import bp

_http_session: Optional[requests.Session] = None

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

app.register_functions(bp) # register the DF functions

def get_http_session() -> requests.Session:
    """Get or create the HTTP session (singleton pattern)"""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Configure session defaults
        _http_session.headers.update({
            'User-Agent': 'Azure-Function-App/1.0',
            'Accept': 'application/json'
        })
        # Set connection pooling and timeout defaults
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        _http_session.mount('http://', adapter)
        _http_session.mount('https://', adapter)
        logging.info("HTTP session initialized")
    
    return _http_session

# @app.route(route="http_trigger_aisearch")
# @openapi(
#     summary="Retrieves supporting documents for a user query",
#     description="Get supporting documentation for a specific architecture.",
#     tags=["AISearch"],
#     operation_id="http_trigger_aisearch",
#     route="/api/http_trigger_aisearch",
#     method="post",
#     request_model=user_query_request,
#     response_model=search_response,
#     response={
#         200: {"description": "Search results retrieved successfully"},
#         400: {"description": "Invalid request"}
#     }
# )
# def http_trigger_aisearch(req: func.HttpRequest) -> func.HttpResponse:
#     request_id = req.headers.get('x-request-id', 'unknown')
#     logging.info(f'[{request_id}] Starting HTTP trigger function - AI Search endpoint called')
    
#     try:
#         logging.info(f'[{request_id}] Step 1: Initializing clients...')
        
#         # Parse request body
#         user_query = None
#         try:
#             req_body = req.get_json()
#             logging.info(f'[{request_id}] Request body parsed successfully:  {json.dumps(req_body)}')
            
#             if req_body:
#                 user_query = req_body.get('user_query')  # Use dict.get() method
#                 logging.info(f'[{request_id}] User query extracted: "{user_query}"')
#             else:
#                 logging.warning(f'[{request_id}] Request body is empty or invalid')
#                 raise ValueError("Request body is empty or invalid")
#         except ValueError as ve:
#             logging.error(f'[{request_id}] JSON parsing error: {str(ve)}')
#             pass

#         if user_query:
#             logging.info(f'[{request_id}] Step 3: Starting AI Search pipeline for query: "{user_query}"')
            
#             # Perform AI Search using the initialized client
#             try:
                
#                 search_results = execute_ai_search(request_id, user_query)
#                 logging.info(f'[{request_id}] ✅ Search pipeline completed successfully')
#                 return func.HttpResponse(
#                     body=json.dumps(search_results),
#                     status_code=200,
#                     mimetype="application/json"
#                 )

#             except Exception as search_error:
#                 logging.error(f'[{request_id}] ❌ Search pipeline error in step 3: {str(search_error)}', exc_info=True)
#                 return func.HttpResponse(
#                     f"Search failed: {str(search_error)}",
#                     status_code=500
#                 )
        
#         else:
#             logging.warning(f'[{request_id}] No user_query provided in request')
#             return func.HttpResponse(
#                 "This HTTP triggered function executed successfully. Pass a 'user_query' in the query string or request body for a personalized search response.",
#                 status_code=400
#             )
            
#     except Exception as e:
#         logging.error(f'[{request_id}] ❌ Function execution error: {str(e)}', exc_info=True)
#         return func.HttpResponse(
#             f"Function execution failed: {str(e)}",
#             status_code=500
#         )

@app.route(route="get_job_result", methods=["POST"])  # <-- Ensure POST method
@openapi(
    summary="Retrieves the result of a long-running job",
    description="Fetches the result of a job using the provided 'statusQueryGetUri'.",
    tags=["JobResults"],
    operation_id="get_job_result",
    route="/api/get_job_result",
    method="post",  # <-- Change to POST
    request_model=status_get_uri_request,
    response_model=status_get_uri_response,
    response={
        200: {"description": "Job result retrieved successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
def get_job_result(req: func.HttpRequest) -> func.HttpResponse:
    """
    This function takes the 'statusQueryGetUri' value and returns the raw response from it to the caller
    """
    request_id = req.headers.get('x-request-id', 'unknown')
    logging.info(f'[{request_id}] Starting HTTP trigger function - AI Search endpoint called')
    
    # For POST, get parameter from body
    try:
        req_body = req.get_json()
        status_query_get_uri = req_body.get('statusQueryGetUri') if req_body else None
    except Exception:
        status_query_get_uri = None

    if not status_query_get_uri:
        logging.error("No 'statusQueryGetUri' provided in the request")
        return func.HttpResponse(
            "Please provide a 'statusQueryGetUri' parameter.",
            status_code=400
        )
    logging.info(f"Received 'statusQueryGetUri': {status_query_get_uri}")
    try:
        http_session = get_http_session()
        logging.info(f'[{request_id}] ✅ HTTP session initialized successfully')
        # Make a GET request to the provided URI
        response = http_session.get(status_query_get_uri)
        response.raise_for_status()  # Raise an error for bad responses
        
        result_json = response.json()
        status = result_json.get('runtimeStatus') or result_json.get('customStatus')

        if status and status.lower() == 'completed':
            output = result_json.get('output')
            return func.HttpResponse(
                body=json.dumps({
                    "status": "Completed",
                    "results": output
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                body=json.dumps({
                    "status": status or "Unknown"
                }),
                status_code=200,
                mimetype="application/json"
            )
    except Exception as e:
        logging.error(f"Error fetching job result: {e}")
        return func.HttpResponse(
            "Unexpected error occurred.",
            status_code=500
        )

@app.route(route="openapi.json")
def openapi_spec(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(get_openapi_json())

@app.route(route="docs")
def swagger_ui(req: func.HttpRequest) -> func.HttpResponse:
    return render_swagger_ui()

@app.queue_trigger(arg_name="msg", queue_name="inputuserquery", connection="STORAGE_CONNECTION")
@app.queue_output(arg_name="outputQueue", queue_name="outputresults", connection="STORAGE_CONNECTION")  
def queue_trigger_aisearch(msg: func.QueueMessage, outputQueue: func.Out[str]):
    try:
        logging.info('Starting queue trigger function - AI Search pipeline triggered')
        logging.info(f'The function receives the following message from the input queue: {msg.get_body().decode("utf-8")}')
        # Parse the message payload
        logging.info('Parsing message payload...')
        if not msg.get_body():
            logging.error('Input queue message is empty')
            raise ValueError("Input queue message is empty")
        logging.info('Decoding message payload...')
        
        inputQueueMessageId = msg.id
        logging.info(f'Input queue message ID: {inputQueueMessageId}')
        messagepayload = json.loads(msg.get_body().decode("utf-8"))
        logging.info(f'The function receives the following message: {json.dumps(messagepayload)}')
        user_query = messagepayload["user_query"]
        
        response_message = execute_ai_search(inputQueueMessageId, user_query)
        
        logging.info(f'The function returns the following message through the output queue: {json.dumps(response_message)}')

        outputQueue.set(json.dumps(response_message))

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        