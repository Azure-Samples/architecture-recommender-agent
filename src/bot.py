import os
import sys
import traceback
import json
from typing import Any, Dict, Optional
from dataclasses import asdict
from botbuilder.core import MemoryStorage, TurnContext, Middleware
from state import AppTurnState
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.ai.actions import ActionTurnContext
from teams.state import TurnState
from teams.feedback_loop_data import FeedbackLoopData
import aiohttp
import logging
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # This ensures output goes to terminal
    ]
)

logger = logging.getLogger(__name__)

from config import Config

config = Config()


# Use the no-auth config instead of your regular config
adapter = TeamsAdapter(config)

project = AIProjectClient(
  endpoint=config.FOUNDRY_PROJECT_ENDPOINT,  # Replace with your endpoint
  credential=DefaultAzureCredential())  # Replace with your project key)
agent = project.agents.get_agent(config.FOUNDRY_AGENT_ID)

# Define storage and application
storage = MemoryStorage()
bot_app = Application[AppTurnState](
    ApplicationOptions(
        bot_app_id="",
        storage=storage,
        adapter=adapter
    )
)


@bot_app.turn_state_factory
async def turn_state_factory(context: TurnContext):
    return await AppTurnState.load(context, storage)

@bot_app.activity("message")
async def on_message(context: TurnContext, state: AppTurnState):
    user_input = context.activity.text or ""
    logger.info(f"Received user input: {user_input}")
    
    # Get thread ID from state
    thread_id = getattr(state.conversation, 'foundry_thread_id', None)

    if thread_id is not None:

        logger.info(f"Using existing thread ID: {thread_id}")
        
    else:
        logger.info("No existing thread ID found - starting new conversation")
        thread = project.agents.threads.create()
        thread_id = thread.id
        state.conversation.foundry_thread_id = thread_id
        logger.info(f"Created new thread with ID: {thread_id}")

    try:
        existing_runs = list(project.agents.runs.list(thread_id=thread_id))
        active_runs = [r for r in existing_runs if r.status in ["queued", "in_progress", "requires_action"]]
        
        if active_runs:
            logger.info(f"Found {len(active_runs)} active run(s). Waiting for completion...")
            
            # Wait for existing runs to complete
            for active_run in active_runs:
                try:
                    completed_run = project.agents.runs.get_and_process(
                        thread_id=thread_id, 
                        run_id=active_run.id
                    )
                    messages = list(project.agents.messages.list(thread_id=thread_id))
                    assistant_response = messages[0].content[0].text.value if messages[0].role == "assistant" else "No assistant response found."
                    logger.info(f"Run {active_run.id} finished with status: {completed_run.status}")
                    context.send_activity(assistant_response)
                            
                except Exception as wait_error:
                    logger.error(f"Error waiting for run {active_run.id}: {wait_error}")
                    # Try to cancel the problematic run
                    try:
                        project.agents.runs.cancel(thread_id=thread_id, run_id=active_run.id)
                    except:
                        pass
            
            # Create new run for the current message
            logger.info("Creating new run to process current message...")
            
            message = project.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input)
            
            run = project.agents.runs.create_and_process(
                thread_id=thread_id,
                agent_id=agent.id)
        else:
            # No active runs, create new run
            logger.info("No active runs found, creating new run...")
            message = project.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input)

            run = project.agents.runs.create_and_process(
                thread_id=thread_id,
                agent_id=agent.id
            )
            
    except Exception as run_error:
        logger.error(f"Error handling runs: {run_error}")
        await context.send_activity("Sorry, I'm having trouble processing your request. Please try again.")
        return
    
    # Get the assistant's messages from the thread
    messages = list(project.agents.messages.list(thread_id=thread_id))
    logger.info(f"Messages: {messages}")

    assistant_response = messages[0].content[0].text.value if messages[0].role == "assistant" else "No assistant response found."
    
    # Save thread ID for next turn
    state.conversation.foundry_thread_id = thread_id
    
    # Send response back to user
    await context.send_activity(assistant_response)

    
@bot_app.error
async def on_error(context: TurnContext, error: Exception):
    # This check writes out errors to console log .vs. app insights.
    # NOTE: In production environment, you should consider logging this to Azure
    #       application insights.
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Send a message to the user
    await context.send_activity("The agent encountered an error or bug.")
