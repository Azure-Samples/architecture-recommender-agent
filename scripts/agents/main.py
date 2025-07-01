import asyncio
import logging
import os

from intake_agent import IntakeAgent
from researcher_agent import ArchitectureResearcherAgent
from agent_factory import agent_factory
from utils import initialize_telemetry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_agent_flow():
    try:
        # Initialize factory and create agents
        logger.info("Initializing agent factory and creating agents...")
        factory = await agent_factory.get_factory()
        
        # Create both agents
        researcher_agent = await factory.create_agent(ArchitectureResearcherAgent)
        intake_agent = await factory.create_agent(IntakeAgent)
        
        
        # Set connected agents
        intake_agent.set_connected_agents({
            "researcher": researcher_agent
        })
        
        # Run test query
        logger.info("Running test query through intake agent...")
        user_query = "What architecture would you recommend for a high-scale e-commerce platform?"
        response = await intake_agent.query(user_query)
        
        logger.info("Final Agent Response:")
        print(response["assistant_response"])
    
    except Exception as e:
        logger.error(f"An error occurred during agent flow: {str(e)}")
    
    finally:
        # Cleanup agents and factory
        logger.info("Cleaning up agents and factory...")
        # await agent_factory.cleanup()

if __name__ == "__main__":
    # Initialize telemetry
    logger.info("Initializing telemetry...")
    initialize_telemetry()
    logger.info("Starting agent flow...")
    asyncio.run(run_agent_flow())
