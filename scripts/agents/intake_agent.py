# Azure AI Agent for software architecture intake using Azure AI Projects SDK

import os
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from dotenv import load_dotenv
from azure.ai.agents.models import FunctionTool, ConnectedAgentTool

from agent_factory import BaseAgent
from core_prompts import INTAKE_AGENT_PROMPT

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntakeAgent(BaseAgent):
    """Azure AI Agent for recommending software architectures based on user requirements.
    
    This is the main orchestrator agent that can call connected researcher and summarizer agents.
    """
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
        mock_architecture_data = {"architecture": "Microservices using Azure Container Apps", "citations": ["https://example.com/microservices"]}
        
        return json.dumps(mock_architecture_data)

    def __init__(self, factory):
        super().__init__(factory)
        self.connected_agents: Dict[str, BaseAgent] = {}
       # self.functions = FunctionTool(functions={self.fetch_architecture_recommendation})

    def get_agent_name(self) -> str:
        """Return the name for this agent."""
        return "Software Architecture Intake Agent (Main Orchestrator)"
    
    def set_connected_agents(self, connected_agents: ConnectedAgentTool):
        """Set the connected agents that this intake agent can call."""
        
        # Combine existing tools with connected agent tools
        all_tools =  connected_agents.definitions
        
        # Update the agent with the new tools
        tool_agent = self.factory.client.agents.update_agent(
            agent_id=self.agent_id,
            tools=all_tools,
            headers={"x-ms-enable-preview": "true"}
        )
        self.agent_definition = tool_agent
       
    
    
    def get_required_function_tools(self) -> Optional[FunctionTool]:
        """Return required function tools for the researcher agent."""
        return None 
    
    def get_required_tools(self) -> List[str]:
        """Return required tools for the intake agent.
        
        The intake agent coordinates with other agents and doesn't directly search.
        It uses function calling to coordinate with research and summarizer agents.
        
        Returns:
            List containing only function tools (no AI search needed)
        """
        return []  # Only function tools, no AI search needed
    
    def get_agent_instructions(self) -> str:
        """Get the system instructions for the agent."""
        return INTAKE_AGENT_PROMPT

    async def query(self, user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and return the agent's response.
        Uses the base class functionality with function calling support.
        
        Args:
            user_query: The user's question or request
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Use the base class query method which handles function calling
        return await super().query(user_query, thread_id)

if __name__ == "__main__":
    import sys
    from agent_factory import agent_factory
    from researcher_agent import ArchitectureResearcherAgent

    async def main():
        try:
            # Initialize factory
            factory = await agent_factory.get_factory()

            # Create agents
            
            researcher_agent = await factory.create_agent(ArchitectureResearcherAgent)
            
            connected_agent = ConnectedAgentTool(
               id=researcher_agent.agent_id, name="ArchitectureResearchAgent", description="Researches software architecture patterns"
            )

            intake_agent = await factory.create_agent(IntakeAgent)

            intake_agent.set_connected_agents(connected_agent)
           
            
            # Example query
            user_query = "Can you recommend an architecture for a large-scale social media app?"
            logger.info("Sending query to intake agent...")
            result = await intake_agent.query(user_query)
            
            logger.info("Agent Response:")
            print(json.dumps(result, indent=2))

        except Exception as e:
            logger.error(f"Error in main: {e}")
            sys.exit(1)
        finally:
            logger.info("Cleaning up...")
            #await agent_factory.cleanup()

    asyncio.run(main())