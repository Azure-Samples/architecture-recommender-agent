from pydantic import BaseModel
from typing import List, Optional

class user_query_request(BaseModel):
    """
    Model for user query request.
    """
    user_query: str
    
class document(BaseModel):
    id: str
    name: str
    architecture_url: Optional[str] = ""
    content: str
    score: float  # Changed from str to float
    
class search_response(BaseModel):
    documents: List[document]

class orchestration_status_response(BaseModel):
    """
    Model for durable function orchestration status response.
    """
    id: str
    statusQueryGetUri: str
    sendEventPostUri: str
    terminatePostUri: str
    rewindPostUri: str
    purgeHistoryDeleteUri: str
    restartPostUri: str
    suspendPostUri: str
    resumePostUri: str

class status_get_uri_request(BaseModel):
    statusQueryGetUri: str

class status_get_uri_response(BaseModel):
    status: str
    results: Optional[dict] = None

class api_response(BaseModel):
    """
    Model for API response.
    """
    name: str
    instanceId: str
    runtimeStatus: str
    input: Optional[str] = None
    customStatus: Optional[str] = None
    output: Optional[str] = None