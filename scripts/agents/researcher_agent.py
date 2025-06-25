# Azure AI Agent for researching and analyzing software architecture patterns

import json
import logging
from typing import Dict, Any, Optional, List

from azure.ai.agents.models import FunctionTool
from agent_factory import BaseAgent
from core_prompts import RESEARCHER_AGENT_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchitectureResearcherAgent(BaseAgent):
    """Azure AI Agent for researching detailed architecture patterns and technologies."""
    def __init__(self, factory):
        super().__init__(factory)
    
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
        return None # No function tools required

    def get_agent_instructions(self) -> str:
        """Get the system instructions for the agent."""
        return RESEARCHER_AGENT_PROMPT

    async def query(self, user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and force use of AI Search.
        """
        # Ensure run creation forces tool use
        run = self.factory.client.agents.runs.create_and_process(
            thread_id=thread_id,
            agent_id=self.agent_id,
            tool_choice="required"  # <-- force use of AI Search
        )

        # Wait for completion and get result
        completed_run = self.factory.client.agents.runs.get_and_process(
            thread_id=thread_id,
            run_id=run.id
        )

        messages = list(self.factory.client.agents.messages.list(thread_id=thread_id))
        
        # Extract assistant message
        assistant_response = next(
            (msg for msg in messages if msg.role == "assistant"),
            None
        )

        if assistant_response:
            content = assistant_response.content[0].text.value
            result = {
                "assistant_response": content,
                "status": completed_run.status
            }
        else:
            result = {
                "assistant_response": "No response generated.",
                "status": "error"
            }

        # Check grounding as before
        if not self._is_grounded_in_search(result["assistant_response"]):
            logger.warning("Response may not be grounded in AI Search results.")
            result["assistant_response"] = (
                "I don't have enough information to answer that based on the current knowledge base."
            )
            result["status"] = "warning"

        return result

    def _is_grounded_in_search(self, response: str) -> bool:
        """Determine if response cites AI Search sources."""
        # Simple heuristic: look for mention of source, citation, or AI Search marker
        markers = ["Source:", "Reference:", "Cited:", "AI Search"]
        if any(marker in response for marker in markers):
            return True
        if "I don't have enough information" in response:
            return True
        return False