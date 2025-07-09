import logging
import azure.functions as func
import azure.durable_functions as df
from ai_search_logic import execute_ai_search
from models.models import user_query_request, document, search_response, orchestration_status_response
from azure_functions_openapi.decorator import openapi

# To learn more about blueprints in the Python prog model V2,
# see: https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python?tabs=asgi%2Capplication-level&pivots=python-mode-decorators#blueprints

# Note, the `func` namespace does not contain Durable Functions triggers and bindings, so to register blueprints of
# DF we need to use the `df` package's version of blueprints.
bp = df.Blueprint()

# We define a standard function-chaining DF pattern

@bp.route(route="startOrchestrator")
@bp.durable_client_input(client_name="client")
@openapi(
    summary="Start a durable orchestration for AI Search",
    description="Initiates a durable function orchestration to perform AI search and document retrieval.",
    tags=["DurableFunctions"],
    operation_id="start_orchestrator",
    route="/api/startOrchestrator",
    method="post",
    request_model=user_query_request,
    response_model=orchestration_status_response,
    response={
        200: {"description": "Orchestration started successfully"},
        400: {"description": "Invalid request"}
    }
)
async def start_orchestrator(req: func.HttpRequest, client):
    try:
        # Get the input from the request body
        req_body = req.get_json()
        logging.info(f'Request body received: {req_body}')
        
        if not req_body:
            return func.HttpResponse(
                "Request body is required",
                status_code=400
            )
        
        input_data = req_body.get('user_query') if req_body else None
        
        if not input_data:
            return func.HttpResponse(
                "user_query is required in request body",
                status_code=400
            )
        
        instance_id = await client.start_new("my_orchestrator", None, input_data)
        
        logging.info(f"Started orchestration with ID = '{instance_id}'.")
        return client.create_check_status_response(req, instance_id)
        
    except Exception as e:
        logging.error(f"Error starting orchestration: {str(e)}")
        return func.HttpResponse(
            f"Failed to start orchestration: {str(e)}",
            status_code=500
        )

@bp.orchestration_trigger(context_name="context")
def my_orchestrator(context: df.DurableOrchestrationContext):
    logging.info("Orchestrator started.")
    # Get the input passed from the starter function
    user_query = context.get_input()
    logging.info(f"Processing query: {user_query}")
    result = yield context.call_activity('retrieve_documents', user_query)
    return result

@bp.activity_trigger(input_name="user_query")
def retrieve_documents(user_query: str) -> dict:
    logging.info(f"Retrieving documents for query: {user_query}")
    # Call the AI search logic to get documents
    results = execute_ai_search("activity", user_query)
    logging.info(f"Retrieved documents for query: {user_query}")
    return results  # Already a dictionary, JSON serializable