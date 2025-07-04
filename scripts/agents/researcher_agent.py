# Azure AI Agent for researching and analyzing software architecture patterns

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List,Set
from azure.search.documents.models import VectorizedQuery
from azure.ai.agents.models import FunctionTool,OpenApiAnonymousAuthDetails, OpenApiTool, AzureFunctionTool,AzureFunctionStorageQueue,ToolSet,SubmitToolOutputsAction,RequiredFunctionToolCall,ToolOutput,ThreadRun
from agent_factory import BaseAgent
from core_prompts import RESEARCHER_AGENT_PROMPT
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from azure.core.exceptions import AzureError
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NUM_SEARCH_RESULTS = 15
K_NEAREST_NEIGHBORS = 30

_factory = None
_agent_id = None




class ArchitectureResearcherAgent(BaseAgent):
    """Azure AI Agent for researching detailed architecture patterns and technologies."""

    def __init__(self, factory):
        super().__init__(factory)
        global _factory, _agent_id
        _factory = factory
        _agent_id = self.agent_id
        self.credential = DefaultAzureCredential()
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
        self.ai_client = AIProjectClient(
                endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "https://demojul25.services.ai.azure.com/api/projects/demoproject"),
                credential=self.credential
            )

        with open(os.path.join(os.path.dirname(__file__), "ai-search-fn-api.json"), "r") as f:
            openapi_aisearch = json.loads(f.read())
            
        openapi_aisearch["servers"] = [ { "url": os.environ.get("AZURE_AISEARCH_FUNCTION_TOOL_URI", "") } ]

        print(f"Using AI Search Function Tool URI: {openapi_aisearch['servers'][0]['url']}")
        
        # Create Auth object for the OpenApiTool (note: using anonymous auth here; connection or managed identity requires additional setup)
        auth = OpenApiAnonymousAuthDetails()

        # Initialize the main OpenAPI tool definition for supporting documentation
        openapi_tool = OpenApiTool(
            name="GetSupportingDocumentation", spec=openapi_aisearch, description="Use to retrieve supporting documentation to assist in recommending software architectures.", auth=auth
        )
        
        # azure_function_tool = AzureFunctionTool(
        #     name="GetSupportingDocumentation",
        #     description="Get supporting documentation for a specific architecture.",
        #     parameters={
        #         "type": "object",
        #         "properties": {
        #             "user_query": { "type": "string", "description": "The user query to look up." },
        #         },
        #         "required": [ "user_query" ],
        #     },
        #     input_queue=AzureFunctionStorageQueue(
        #         queue_name="inputuserquery",
        #         storage_service_endpoint=storage_connection_string,
        #     ),
        #     output_queue=AzureFunctionStorageQueue(
        #         queue_name="outputresults",
        #         storage_service_endpoint=storage_connection_string
        #     )
        # )
       # agent_researcher_default_functions: Set = {
       #         query,
       #     }

        #agent_researcher_default_functions = FunctionTool(functions=agent_researcher_default_functions)
        #toolset = ToolSet()
        #toolset.add(agent_researcher_default_functions)
        #self.functions = toolset
        #factory.client.agents.enable_auto_function_calls(toolset=self.functions)
        #self.functions = azure_function_tool
        self.functions = openapi_tool
        #AzureFunctionTool(functions={self.fetch_architecture_recommendation})
           
    def get_agent_name(self) -> str:
        """Return the name for this agent."""
        return "Architecture Research Agent"
    
    def get_required_tools(self) -> List[str]:
        """Return required tools for the researcher agent.
        
        The researcher agent needs AI search capabilities to access the knowledge base
        and perform deep research on architecture patterns and technologies.
        
        Returns:
            List containing 'ai_search' for knowledge base access        """
        return None  # Only function tools, no AI search needed
        #return ['ai_search']
    
    def get_required_function_tools(self) -> Optional[FunctionTool]:
        """Return required function tools for the researcher agent."""
        return self.functions
        
    def get_agent_instructions(self) -> str:
        """Get the system instructions for the agent."""
        return RESEARCHER_AGENT_PROMPT
    
    # async def aisearch_query(self, user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    #     """
    #     Perform a search using Azure Cognitive Search with both semantic and vector queries.
    #     Searches across name, Content, and Architecture URL fields.
        
    #     Args:
    #         search_query: The user's search query
            
    #     Returns:
    #         List of search results with combined content
    #     """
    #     try:
    #         print(f"🔍 Running search for: '{user_query}'")
            
    #         # Generate vector embedding for the query
    #         query_vector = self.embeddings_model.embed_query(user_query)
            
    #         # Create vector queries for all three vector fields
    #         vector_queries = [
    #             VectorizedQuery(
    #                 vector=query_vector,
    #                 k_nearest_neighbors=K_NEAREST_NEIGHBORS,
    #                 fields="content_vector"
    #             )
    #         ]
            
    #         # Perform the search with all vector fields and corresponding text fields
    #         results = self.search_client.search(
    #             search_text=user_query,
    #             vector_queries=vector_queries,
    #             select=["id", "name", "architecture_url", "content"],
    #             top=NUM_SEARCH_RESULTS
    #         )
            
    #         search_results = []
    #         for result in results:
    #             # Combine all text content for the LLM with clear delineation
    #             content_parts = []
                
    #             # Always include title at the top
    #             if result.get("name"):
    #                 content_parts.append(f"=== NAME ===\n{result['name']}\n=== END NAME ===")
                
    #             if result.get("architecture_url"):
    #                 content_parts.append(f"=== URL ===\n{result['architecture_url']}\n=== END URL ===")

    #             if result.get("content"):
    #                 content_parts.append(f"=== CONTENT ===\n{result['content']}\n=== END CONTENT ===")
                
    #             combined_content = "\n\n".join(content_parts)
                
    #             search_result = {
    #                 "id": result["id"],
    #                 "name": result['name'],
    #                 "architecture_url": result.get("architecture_url", ""),
    #                 "content": combined_content,
    #                 "score": result["@search.score"]
    #             }
    #             search_results.append(search_result)
    #         final_answer= await self.generate_answer(user_query, combined_content)

    #         print(f"✅ Found {len(search_results)} search results")
    #         print(f"🤖 Generated answer: {final_answer}")

    #         return final_answer

    #     except Exception as e:
    #         print(f"❌ Error during search: {e}")
    #         raise
    
    async def generate_answer(self, user_query: str, ai_search_result: Dict[str, Any]) -> str:
        """
        Generate an answer using Azure AI Foundry agent and ai search results.
        
        Args:
            user_query: The user query passed to the ai search tool 
            ai_search_result: List of search results from Azure Search for the user query
            
        Returns:
            Generated answer string
        """
        try:
            print(f"🤖 Generating answer for: '{ai_search_result}'")

            # Create a thread for this conversation
            thread = self.ai_client.agents.threads.create()
            
            # Format search results for the agent
            formatted_results = []
          
              # Create the user message with the exact format from document_rag.py
            user_message = f"""Create a comprehensive answer by analyzing the search results.

    AI Search Result: {user_query}

    Search Results:
    {chr(10).join(ai_search_result)}

    Synthesize these results into a clear, complete answer. Remember to cite which documents contain the information you're referencing."""
            user_message = f"""Create a comprehensive answer to the user's question using these search results.

    User Question: {user_query}

    Search Results:
    {chr(10).join(ai_search_result)}

    Synthesize these results into a clear, complete answer. Remember to cite which documents contain the information you're referencing."""

            # Add message to thread
            self.ai_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Run the agent
            run = self.ai_client.agents.runs.create_and_process(
                thread_id=thread.id, 
                agent_id=self.agent_id
            )
            
            # Check if the run failed
            if run.status == "failed":
                raise Exception(f"Agent run failed: {run.last_error}")
            
            # Get the response messages
            messages = self.ai_client.agents.messages.list(thread_id=thread.id)
            
            # Find the latest assistant message
            assistant_message = None
            for message in messages:
                if message.role == "assistant":
                    assistant_message = message
                    break
            
            if not assistant_message:
                raise Exception("No response from agent")
            
            # Extract content from the message
            response_content = ""
            for content_item in assistant_message.content:
                if hasattr(content_item, 'text'):
                    response_content += content_item.text.value
            
            # Clean up thread (optional - could be kept for conversation history)
            try:
                self.ai_client.agents.threads.delete(thread_id=thread.id)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup thread: {cleanup_error}")
            
            print("✅ Answer generated successfully")
            return response_content
            
        except AzureError as e:
            print(f"❌ Azure error during answer generation: {e}")
            raise
        except Exception as e:
            print(f"❌ Error during answer generation: {e}")
            raise