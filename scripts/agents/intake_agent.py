# Azure AI Agent for software architecture intake using Azure AI Projects SDK

import os
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List,Set
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
    def __init__(self, factory):
        super().__init__(factory)
        self.connected_agents: Dict[str, BaseAgent] = {}
       

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
    from utils import initialize_telemetry
        
    # Initialize telemetry
    logger.info("Initializing telemetry...")
    initialize_telemetry()
    logger.info("Starting agent flow...")

    async def main():
        try:
            # Initialize factory
            factory = await agent_factory.get_factory()
              
            
            # Create agents
            researcher_agent = await factory.create_agent(ArchitectureResearcherAgent)
            
            #factory.client.agents.enable_auto_function_calls({researcher_agent.query})
            
            connected_agent = ConnectedAgentTool(
               id=researcher_agent.agent_id, name="ArchitectureResearchAgent", description="After capturing sufficient detail from the user, we use this agent to perform the necessary research on recommended software architecture patterns"
            )

            intake_agent = await factory.create_agent(IntakeAgent)

            intake_agent.set_connected_agents(connected_agent)
           
            
            # Example query
            #user_query = "Data from multiple sources, such as fare data and trip data, is ingested through Event Hubs. These streams are then processed in Azure Databricks,"
            user_query = "I need data from multiple sources, such as fare data and trip data, is ingested through Event Hubs and stored in Azure storage blob. The data will be provided by external partners and will not need to be secured for now as this is a POC. The processing will be triggered only when the file is uploaded and it will go to an SFTP which will be created so we need detail on this as well. These streams are then processed in Azure Databricks where we can report on afterward. What is the recommendation?"
            logger.info("Sending query to intake agent...")
            result = await intake_agent.query(user_query)
            thread = intake_agent.client.agents.threads.create()
            researcher_response = await researcher_agent.query(user_query, thread_id=thread.id)
            logger.info("Researcher Agent Response:")
            print(json.dumps(researcher_response, indent=2))
            logger.info("Agent Response:")
            print(json.dumps(result, indent=2))

        except Exception as e:
            logger.error(f"Error in main: {e}\r\n{e.__traceback__}")
            sys.exit(1)
        finally:
            logger.info("Cleaning up...")
            #await agent_factory.cleanup()

    asyncio.run(main())