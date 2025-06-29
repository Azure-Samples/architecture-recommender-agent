import os
import sys
import traceback
import json
from typing import Any, Dict, Optional, List
from dataclasses import asdict
from botbuilder.core import MemoryStorage, TurnContext, Middleware
from state import AppTurnState
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.feedback_loop_data import FeedbackLoopData
import aiohttp
import logging
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from msgraph import GraphServiceClient
from msgraph.generated.users.item.member_of.member_of_request_builder import MemberOfRequestBuilder
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # This ensures output goes to terminal
    ]
)

logger = logging.getLogger(__name__)

config = Config()

# Use the no-auth config instead of your regular config
adapter = TeamsAdapter(config)

# Initialize AIProjectClient with your endpoint and credentials. For credentials, service principal can be used if .env has following configured: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
project = AIProjectClient(
  endpoint=config.FOUNDRY_PROJECT_ENDPOINT,  # Replace with your endpoint
  credential=DefaultAzureCredential())  # Replace with your project key)
agent = project.agents.get_agent(config.FOUNDRY_AGENT_ID)


# Define application
storage = MemoryStorage() # this is used to store conversation state, user state, etc. It can be replaced with other storage options like CosmosDB.
bot_app = Application[AppTurnState](
    ApplicationOptions(
        bot_app_id="",
        storage=storage,
        adapter=adapter
    )
)

async def check_agent_app_access(user_aad_id: str, authorized_group: str) -> bool:
    """
    Check if a user has access to the specific AI Foundry project.
    
    Args:
        user_aad_id: The AAD object ID of the user
        
    Returns:
        bool: True if user has access, False otherwise
    """
    try:
        credential = DefaultAzureCredential()

        graph_client = GraphServiceClient(
            credentials=credential,
            scopes=['https://graph.microsoft.com/.default']
        )

        # Get the user's group memberships
        member_of = await graph_client.users.by_user_id(user_aad_id).member_of.get()

        # Define the specific Entra ID groups that should have access
        authorized_group = authorized_group
        has_access = False

        # Check if the user is a member of the Required Agent App group
        user_groups = []
        if member_of and member_of.value:
            for group in member_of.value:
                # Check if it's a group and has a display name
                if hasattr(group, 'display_name') and group.display_name:
                    user_groups.append(group.display_name)
                    logger.info(f"User {user_aad_id} is member of group: {group.display_name}")

                    if group.display_name == authorized_group:
                        has_access = True
                        logger.info(f"User {user_aad_id} has access via group membership: {authorized_group}")

        if not has_access:
            logger.warning(f"User {user_aad_id} is not a member of authorized group '{authorized_group}'. User groups: {user_groups}")

        return has_access

    except Exception as e:
        logger.error(f"Error checking project access: {e}")
        return False


@bot_app.turn_state_factory
async def turn_state_factory(context: TurnContext):
    return await AppTurnState.load(context, storage)

@bot_app.activity("message")
async def on_message(context: TurnContext, state: AppTurnState):
    user_input = context.activity.text or ""
    logger.info(f"Received user input: {user_input}")

    # Get user aad id from state
    user_aad_id = getattr(state.conversation, 'user_aad_id', None) 

    if user_aad_id is None:
        # If AAD object ID is not set, try to get it from the activity
        logger.info("No user AAD ID found in state, checking activity...")
        user_aad_id = getattr(context.activity.from_property, 'aad_object_id', None)
        if user_aad_id:
            state.conversation.user_aad_id = user_aad_id
            logger.info(f"Set user AAD ID in state: {user_aad_id}")
            has_access = await check_agent_app_access(user_aad_id=user_aad_id, authorized_group=config.FOUNDRY_AGENT_APP_GROUP_NAME)
            #has_access = await check_agent_app_access(user_aad_id="", authorized_group="")  # Replace with your actual group name
            if not has_access:
                logger.warning(f"User {user_aad_id} does not have access to the Agent App.")
                await context.send_activity("You do not have access to either the required Agent App or the project. Please contact your administrator.")
                return
        else:
            logger.warning("No AAD object ID found in activity or state. Assuming no access to the required Agent App.")
            await context.send_activity("You do not have access to either the required Agent App or the project. Please contact your administrator.")
            return

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
