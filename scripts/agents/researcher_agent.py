# Azure AI Agent for researching and analyzing software architecture patterns

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List,Set
from azure.search.documents.models import VectorizedQuery
from azure.ai.agents.models import FunctionTool,ToolSet,SubmitToolOutputsAction,RequiredFunctionToolCall,ToolOutput,ThreadRun
from agent_factory import BaseAgent
from core_prompts import RESEARCHER_AGENT_PROMPT
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_REVIEW_ATTEMPTS = 3
MIN_QUALITY_SCORE = 0.7 
NUM_SEARCH_RESULTS = 15
K_NEAREST_NEIGHBORS = 30

_factory = None
_agent_id = None

async def query(user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve software architecture recommendations from knowledge base (AI Search).
    
    :param user_query (str): The user's query with full context to assist in retrieving guidance on recommended software architecture.
    :param thread_id (Optional[str]): Optional thread ID for the conversation context.
    """
    attempt = 0
    best_result = None
    best_score = 0.0
    search_results = None

    while attempt < MAX_REVIEW_ATTEMPTS:
        # Ensure run creation forces tool use
        attempt += 1
        logger.info(f"[Attempt {attempt}] Query: {user_query}")

        run = _factory.client.agents.runs.create_and_process(
            thread_id=thread_id,
            agent_id=_agent_id,
            tool_choice="required"  # <-- force use of AI Search
        )

        # Wait for completion and get result
        completed_run = await _wait_for_run_completion_async(
                        _factory.client,
                        thread_id=thread_id,
                        run_id=run.id
                    )

        messages = list(_factory.client.agents.messages.list(thread_id=thread_id))
            
        assistant_msg = next((m for m in messages if m.role == "assistant"), None)
        
        if assistant_msg:
            content = assistant_msg.content[0].text.value
            thought_process = _extract_thought_process(content)
            result = {
                        "assistant_response": content,
                        "thought_process": thought_process,
                        "status": completed_run.status
                    }

            score = _review_response(content, user_query, search_results)
            logger.info(f"[Score: {score}] Thought: {thought_process}")

            if score > best_score:
                best_result = result
                best_score = score

            if score >= MIN_QUALITY_SCORE:
                return best_result

    return best_result or {
        "assistant_response": "Unable to generate a satisfactory response after multiple attempts.",
        "architecture_url": "",
        "thought_process": "Review failed to ground response in AI Search.",
        "status": "error"
        }

def _extract_thought_process(response: str) -> str:
    """Extracts and logs key insight from the AI Search grounded response."""
    try:
        data = json.loads(response)
        return f"Evaluated based on: {data.get('architecture_url', 'No source URL')}"
    except Exception as e:
        logger.warning(f"Could not parse response for thought process: {e}")
        return "Could not extract thought process."
    

def _review_response(response: str, query: str, results: Optional[List[Dict]]) -> float:
    if not results or not response:
        return 0.0
    top = results[0]
    content = top.get("content", "").lower()
    name = top.get("name", "").lower()
    url = top.get("architecture_url", "")
    score = float(top.get("@search.score", 0.0))  # Optional fallback

    response_lower = response.lower()
    response_words = set(response_lower.split())
    content_words = set(content.split())
    name_in_response = name in response_lower
    url_in_response = url and url in response

    # Calculate overlap ratio
    overlap_ratio = len(response_words & content_words) / max(len(content_words), 1)

    # Weighted scoring
    final_score = (
        0.4 * overlap_ratio +         # Response grounded in content
        0.3 * int(name_in_response) + # Mentions document name
        0.2 * int(url_in_response) +  # Cites URL
        0.1 * min(score, 1.0)         # Optional vector score
    )
    return round(final_score, 2)

async def _wait_for_run_completion_async(client, thread_id, run_id, timeout=60, poll_interval=2):
    start_time = asyncio.get_event_loop().time()
    while True:
        run = await asyncio.to_thread(client.agents.runs.get, thread_id=thread_id, run_id=run_id)
        if run.status in ["completed", "failed", "cancelled"]:
            return run
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds.")
        await asyncio.sleep(poll_interval)


class ArchitectureResearcherAgent(BaseAgent):
    """Azure AI Agent for researching detailed architecture patterns and technologies."""

    @staticmethod
    def fetch_architecture_recommendation(context: str) -> str:
        """
        After gathering and clarifying all necessary user requirements, call this function to retrieve the architecture recommendation.
        
        Args:
            context (str): A string containing the full context and requirements of the solution for which an architecture recommendation is needed.
        
        Returns:
            str: Architecture recommendation information along with citations as a JSON string.
        
        Note:
            This function should be called only after sufficient user requirements and context have been collected by the intake agent.
            The current implementation simulates fetching architecture recommendations (mocked as weather data for demonstration).
        """
        # Mock architecture data for demonstration purposes
        mock_architecture_data = {"architecture": "Hello I am executed", "citations": [""]}
        
        return json.dumps(mock_architecture_data)
    
    def __init__(self, factory):
        super().__init__(factory)
        global _factory, _agent_id
        _factory = factory
        _agent_id = self.agent_id
        
        self.search_client = SearchClient(
                os.environ.get("AZURE_AI_SEARCH_ENDPOINT", "https://vectordbdemo.search.windows.net"), 
                os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "cw-architectures-index"), 
                AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"])
            )
        self.embeddings_model = AzureOpenAIEmbeddings(
                azure_deployment=os.environ.get("Azure_OpenAI_Embedding_Deployment_Name", "text-embedding-3-large"),
                api_key=os.environ["AZURE_OPENAI_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            )

       # agent_researcher_default_functions: Set = {
       #         query,
       #     }

        #agent_researcher_default_functions = FunctionTool(functions=agent_researcher_default_functions)
        #toolset = ToolSet()
        #toolset.add(agent_researcher_default_functions)
        #self.functions = toolset
        #factory.client.agents.enable_auto_function_calls(toolset=self.functions)
        #self.functions = FunctionTool(functions={self.fetch_architecture_recommendation})
           
    def get_agent_name(self) -> str:
        """Return the name for this agent."""
        return "Architecture Research Agent"
    
    def get_required_tools(self) -> List[str]:
        """Return required tools for the researcher agent.
        
        The researcher agent needs AI search capabilities to access the knowledge base
        and perform deep research on architecture patterns and technologies.
        
        Returns:
            List containing 'ai_search' for knowledge base access        """
        return ['ai_search']
    
    def get_required_function_tools(self) -> Optional[FunctionTool]:
        """Return required function tools for the researcher agent."""
        #return self.functions
        return None

    def get_agent_instructions(self) -> str:
        """Get the system instructions for the agent."""
        return RESEARCHER_AGENT_PROMPT
    
    async def aisearch_query(self, user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a search using Azure Cognitive Search with both semantic and vector queries.
        Searches across name, Content, and Architecture URL fields.
        
        Args:
            search_query: The user's search query
            
        Returns:
            List of search results with combined content
        """
        try:
            print(f"🔍 Running search for: '{user_query}'")
            
            # Generate vector embedding for the query
            query_vector = self.embeddings_model.embed_query(user_query)
            
            # Create vector queries for all three vector fields
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=K_NEAREST_NEIGHBORS,
                    fields="content_vector"
                )
            ]
            
            # Perform the search with all vector fields and corresponding text fields
            results = self.search_client.search(
                search_text=user_query,
                vector_queries=vector_queries,
                select=["id", "name", "architecture_url", "content"],
                top=NUM_SEARCH_RESULTS
            )
            
            search_results = []
            for result in results:
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
                
                search_result = {
                    "id": result["id"],
                    "name": combined_content,
                    "architecture_url": result.get("architecture_url", ""),
                    "content": result.get("content", ""),
                    "score": result["@search.score"]
                }
                search_results.append(search_result)
            
            print(f"✅ Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            raise
    
    