"""
Agent Factory for creating and managing Azure AI Agents with shared resources.

This factory provides a centralized way to create multiple agents (intake, architecture-researcher, summarizer)
that share the same Azure AI Project connection and resources.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import time

from dotenv import load_dotenv
from azure.ai.agents.models import AzureAISearchTool, FunctionTool
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents with common functionality."""
    def __init__(self, factory: 'AgentFactory'):
        self.factory = factory
        self.client = factory.client
        self.agent_definition = None
        self.agent_id: Optional[str] = None
        self.threads: Dict[str, str] = {}
        self._initialized = False

        # Azure configuration from factory
        self.model_deployment_name = factory.model_deployment_name
        print(f"Model deployment name: {self.model_deployment_name}")
        self.search_index_name = factory.search_index_name
        
        # Tools will be assigned based on requirements during initialization
        self.ai_search_tool: Optional[AzureAISearchTool] = None
    
    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the name for this agent."""
        pass
    
    @abstractmethod
    def get_agent_instructions(self) -> str:
        """Return the system instructions for this agent."""
        pass
    
    @abstractmethod
    def get_required_tools(self) -> List[str]:
        """Return a list of required tool types for this agent.
        
        Possible values:
        - 'ai_search': Azure AI Search tool for searching knowledge base
        
        Returns:
            List of required tool type names
        """
        pass
    
    @abstractmethod
    def get_required_function_tools(self) -> Optional[FunctionTool]:
        """Return a list of required function tool names for this agent.
        
        This allows agents to specify specific function tools they need.
        Currently no agents require any function tools.
        
        Returns:
            List of required function tool names (empty for current agents)
        """
        pass
    
    async def initialize(self):
        """Initialize the agent with Azure AI."""
        try:
            logger.info(f"Initializing {self.get_agent_name()}...")
            
            # Get required tools for this agent
            required_tools = self.get_required_tools()
            required_function_tools = self.get_required_function_tools()
            logger.info(f"Agent {self.get_agent_name()} requires tools: {required_tools}")
            logger.info(f"Agent {self.get_agent_name()} requires function tools: {required_function_tools}")
            
            # Assign AI Search tool if required
            if 'ai_search' in required_tools:
                if self.factory.ai_search_tool:
                    self.ai_search_tool = self.factory.ai_search_tool
                    logger.info(f"Assigned AI Search tool to {self.get_agent_name()}")
                else:
                    logger.warning(f"Agent {self.get_agent_name()} requires AI Search tool but it's not available")
            
            # Combine tools based on requirements
            all_tools = []
            all_tool_resources = None
            
            # Add required function tools (currently none for existing agents)
            if required_function_tools:
                logger.info(f"Required function tools for {self.get_agent_name()}: {required_function_tools}")
                
                functools = self.get_required_function_tools()
                if functools:
                    all_tools.extend(functools.definitions)                
            
            # Add AI Search tool if assigned to this agent
            if self.ai_search_tool:
                all_tools.extend(self.ai_search_tool.definitions)
                all_tool_resources = self.ai_search_tool.resources
                logger.info(f"Added AI Search tool to {self.get_agent_name()}")
            
            # Create agent definition
            self.agent_definition = self.client.agents.create_agent(
                model=self.model_deployment_name,
                name=self.get_agent_name(),
                instructions=self.get_agent_instructions(),
                tools=all_tools if all_tools else None,
                tool_resources=all_tool_resources,
                headers={"x-ms-enable-preview": "true"},
            )
            
            self.agent_id = self.agent_definition.id
            self._initialized = True
            logger.info(f"{self.get_agent_name()} initialized successfully with ID: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.get_agent_name()}: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up agent resources."""
        try:
            if self.agent_id and self.client:
                self.client.agents.delete_agent(self.agent_id)
                logger.info(f"{self.get_agent_name()} deleted successfully")
            
            # Clear threads
            self.threads.clear()
            
        except Exception as e:
            logger.error(f"Error during {self.get_agent_name()} cleanup: {str(e)}")
    
    async def query(self, user_query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and return the agent's response.
        
        Args:
            user_query: The user's question or request
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not self._initialized or self.client is None or not self.agent_id:
            raise RuntimeError("Agent not initialized. Call initialize() first and ensure client is set.")
        
        try:
            # Get or create thread
            thread_id = await self._get_or_create_thread(thread_id)
            
            # Add user message to thread
            self.client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_query
            )
            
            # Create run and handle function calling
            run = self.client.agents.runs.create(
                thread_id=thread_id,
                agent_id=self.agent_id
            )
            
            # Poll the run status and handle function calling
            logger.info(f"Starting run for thread_id={thread_id}, run_id={run.id}, initial status={run.status}")
            while run.status in ["queued", "in_progress", "requires_action"]:
                logger.debug(f"Polling run status: run_id={run.id}, status={run.status}")
                time.sleep(1)
                run = self.client.agents.runs.get(thread_id=thread_id, run_id=run.id)
                logger.info(f"Polled run status: run_id={run.id}, status={run.status}")

                if run.status == "requires_action":
                    logger.info(f"Run requires action: run_id={run.id}")
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []

                    for tool_call in tool_calls:
                        logger.info(f"Handling tool call: tool_call_id={tool_call.id}, function_name={getattr(tool_call, 'name', None)}")
                        # Handle function tool calls by delegating to derived classes
                        output = await self._handle_function_call(tool_call)
                        if output:
                            logger.info(f"Tool call handled: tool_call_id={tool_call.id}, output={output}")
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": output
                            })
                        else:
                            logger.warning(f"No output returned for tool_call_id={tool_call.id}")

                    # Submit tool outputs back to the run
                    if tool_outputs:
                        logger.info(f"Submitting tool outputs for run_id={run.id}: {tool_outputs}")
                        self.client.agents.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
            logger.info(f"Run completed: run_id={run.id}, final status={run.status}")
            
            # Get the assistant's messages from the thread
            messages = self.client.agents.messages.list(thread_id=thread_id)
            
            assistant_response = "I'm sorry, I couldn't generate a response. Please try again."
            # Convert ItemPaged to list and get the latest assistant message
            messages_list = list(messages)
            if messages_list:
                # Messages are typically returned in reverse chronological order (newest first)
                for message in messages_list:
                    if message.role == "assistant":
                        # Extract text content from the message
                        for content in message.content:
                            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                assistant_response = content.text.value
                                break
                        break
            
            return {
                "assistant_response": assistant_response,
                "thread_id": thread_id,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "assistant_response": f"An error occurred while processing your request: {str(e)}",
                "thread_id": thread_id,
                "status": "error"
            }
    
    async def _handle_function_call(self, tool_call) -> Optional[str]:
        """Handle function calls by dispatching to the appropriate function."""
        try:
            if hasattr(tool_call, 'function') and tool_call.function:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments
                
                logger.info(f"Calling function: {function_name} with args: {function_args}")
                
                # Parse arguments if they are JSON string
                if isinstance(function_args, str):
                    import json
                    args = json.loads(function_args)
                else:
                    args = function_args or {}
                
                # Look for the function in this agent instance
                if hasattr(self, function_name):
                    func = getattr(self, function_name)
                    if callable(func):
                        # Call the function with the parsed arguments
                        if args:
                            if isinstance(args, dict):
                                result = func(**args)
                            else:
                                result = func(args)
                        else:
                            result = func()
                        
                        logger.info(f"Function {function_name} returned: {result}")
                        return str(result) if result is not None else "Function executed successfully"
                
                logger.warning(f"Function {function_name} not found in agent {self.get_agent_name()}")
                return f"Function {function_name} not implemented"
                
        except Exception as e:
            logger.error(f"Error executing function call: {str(e)}")
            return f"Error executing function: {str(e)}"
        
        return None

    async def _get_or_create_thread(self, thread_id: Optional[str] = None) -> str:
        """Get existing thread or create a new one."""
        if thread_id and thread_id in self.threads:
            return self.threads[thread_id]
        
        # Create new thread
        thread = self.client.agents.threads.create()
        thread_id = thread.id
        self.threads[thread_id] = thread_id
        
        logger.info(f"Created new thread: {thread_id}")
        return thread_id


class AgentFactory:
    """Factory for creating and managing Azure AI Agents with shared resources."""
    
    def __init__(self):
        self.client: Optional[AIProjectClient] = None
        self.ai_search_tool: Optional[AzureAISearchTool] = None
        self._initialized = False
        self._agents: Dict[str, BaseAgent] = {}
        
        # Azure configuration from environment
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.project_connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
        self.model_deployment_name = "gpt-4.1-mini"
        self.search_index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "cw-architectures-index")
        
        if not self.project_connection_string:
            raise ValueError("AZURE_AI_PROJECT_CONNECTION_STRING is required in environment variables")
    
    async def initialize(self):
        """Initialize the factory with Azure AI Project connection and shared resources."""
        try:
            logger.info("Initializing Agent Factory...")
            
            # Create Azure AI Project client with managed identity
            credential = DefaultAzureCredential()
            self.client = AIProjectClient(
                endpoint=self.project_connection_string,
                credential=credential
            )
            
            # Initialize shared AI Search tool
            await self._initialize_shared_search_tool()
            
            self._initialized = True
            logger.info("Agent Factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Factory: {str(e)}")
            raise
    
    async def _initialize_shared_search_tool(self):
        """Initialize the shared AI Search tool for all agents."""
        try:
            ai_search_conn_id = self._find_search_connection()
            
            if not ai_search_conn_id:
                logger.warning("No Azure AI Search connection found. Agents will be created without search capabilities.")
                self.ai_search_tool = None
            else:
                logger.info(f"Found Azure AI Search connection: {ai_search_conn_id}")
                self.ai_search_tool = AzureAISearchTool(
                    index_connection_id=ai_search_conn_id, 
                    index_name=self.search_index_name
                )
                
        except Exception as e:
            logger.error(f"Error initializing shared search tool: {str(e)}")
            self.ai_search_tool = None
    
    def _find_search_connection(self) -> Optional[str]:
        """Find and return the Azure AI Search connection ID."""
        try:
            for connection in self.client.connections.list():
                if connection.type == ConnectionType.AZURE_AI_SEARCH:
                    logger.info(f"Found Azure AI Search connection: {connection.name}")
                    return connection.id
        except Exception as e:
            logger.error(f"Error finding search connection: {str(e)}")
        return None
    
    async def create_agent(self, agent_class: type) -> BaseAgent:
        """Create and initialize an agent of the specified class."""
        if not self._initialized:
            raise RuntimeError("Factory not initialized. Call initialize() first.")
        
        # Create agent instance
        agent = agent_class(self)
        
        # Initialize the agent
        await agent.initialize()
        
        # Store reference
        agent_name = agent.get_agent_name()
        self._agents[agent_name] = agent
        
        return agent
    
    async def create_all_agents(self, agent_classes: Dict[str, type], connected_agents: Optional[Dict[str, BaseAgent]] = None) -> Dict[str, BaseAgent]:
        """Create and initialize all specified agents."""
        if not self._initialized:
            raise RuntimeError("Factory not initialized. Call initialize() first.")
        
        agents = {}
        
        for agent_type, agent_class in agent_classes.items():
            try:
                logger.info(f"Creating {agent_type} agent...")
                agent = await self.create_agent(agent_class)
                agents[agent_type] = agent
                logger.info(f"Successfully created {agent_type} agent")
            except Exception as e:
                logger.error(f"Failed to create {agent_type} agent: {str(e)}")
                # Clean up any agents created so far
                for created_agent in agents.values():
                    await created_agent.cleanup()
                raise
        
        return agents
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an existing agent by name."""
        return self._agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all created agents."""
        return self._agents.copy()
    
    async def cleanup_all_agents(self):
        """Clean up all created agents."""
        for agent in self._agents.values():
            await agent.cleanup()
        self._agents.clear()
        logger.info("All agents cleaned up successfully")
    
    async def cleanup(self):
        """Clean up the factory and all agents."""
        await self.cleanup_all_agents()
        self._initialized = False
        logger.info("Agent Factory cleaned up successfully")


# Singleton instance for global access
_factory_instance: Optional[AgentFactory] = None


async def get_agent_factory() -> AgentFactory:
    """Get or create the global agent factory instance."""
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = AgentFactory()
        await _factory_instance.initialize()
    
    return _factory_instance


async def cleanup_factory():
    """Clean up the global factory instance."""
    global _factory_instance
    
    if _factory_instance:
        await _factory_instance.cleanup()
        _factory_instance = None


# Create a global factory instance for direct import
class GlobalAgentFactory:
    """Global agent factory wrapper for easy import."""
    
    def __init__(self):
        self._factory: Optional[AgentFactory] = None
    
    async def get_factory(self) -> AgentFactory:
        """Get the initialized factory instance."""
        if self._factory is None:
            self._factory = AgentFactory()
            await self._factory.initialize()
        return self._factory
    async def create_agent(self, agent_class: type) -> BaseAgent:
        """Create a single agent using the factory."""
        factory = await self.get_factory()
        return await factory.create_agent(agent_class)
    
    async def create_all_agents(self, agent_classes: Dict[str, type]) -> Dict[str, BaseAgent]:
        """Create all agents using the factory."""
        factory = await self.get_factory()
        return await factory.create_all_agents(agent_classes)
    
    async def cleanup(self):
        """Clean up the factory."""
        if self._factory:
            await self._factory.cleanup()
            self._factory = None


# Global instance for import
agent_factory = GlobalAgentFactory()