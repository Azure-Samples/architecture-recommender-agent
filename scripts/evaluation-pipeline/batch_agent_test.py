import json
import tempfile
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path
import sys
import os
from datetime import datetime
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import AIAgentConverter, AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator, ToolCallAccuracyEvaluator, RelevanceEvaluator, CoherenceEvaluator, CodeVulnerabilityEvaluator, ContentSafetyEvaluator, IndirectAttackEvaluator, FluencyEvaluator
from azure.ai.evaluation import evaluate
from azure.ai.projects.models import ConnectionType
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
print("🚀 Starting batch agent test...")

project = AIProjectClient(
  endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], 
  credential=DefaultAzureCredential())  

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_deployment="gpt-4o-2",
    api_version="2024-12-01-preview",
)

azure_ai_project = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

agent = project.agents.get_agent(os.environ["FOUNDRY_AGENT_ID"])

quality_evaluators = {evaluator.__name__: evaluator(model_config=model_config) for evaluator in [IntentResolutionEvaluator, TaskAdherenceEvaluator, CoherenceEvaluator]}

safety_evaluators = {evaluator.__name__: evaluator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential()) for evaluator in[ContentSafetyEvaluator]}

quality_and_safety_evaluators = {**quality_evaluators, **safety_evaluators}

evaluator_configs = {}

evaluator_configs["coherence"] = {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.response}"
        }
    }

evaluator_configs["intent_resolution"] = {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.response}"
        }
    }

evaluator_configs["task_adherence"] = {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.response}"
        }
    }

evaluator_configs["content_safety"] = {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.response}"
        }
    }

def batch_evaluation(input_file: str):
    """
    Load queries from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        
    Returns:
        Updates the input file with agent responses and evaluation results
    """
    updated_eval_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                data['response'] = invoke_agent(data['query'])
                updated_eval_data.append(data)

        # Overwrite the original file with updated data
        with open(input_file, 'w', encoding='utf-8') as f:
            for data in updated_eval_data:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully updated {len(updated_eval_data)} entries in {input_file}")
        
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        print(f"Error loading queries: {e}")
        raise


def invoke_agent(query: str) -> str:
    """
    Invoke the agent with a query.
    
    Args:
        query: The query string to send to the agent
        
    Returns:
        Dictionary containing the agent response
    """
    try:
        thread = project.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        message = project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
        )
        print(f"Created message, ID: {message.id}")

        run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print(f"Run finished with status: {run.status}")
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")
        print(f"Run ID: {run.id}")

        messages = list(project.agents.messages.list(thread_id=thread.id))

        assistant_response = messages[0].content[0].text.value if messages[0].role == "assistant" else "No assistant response found."

        return assistant_response        
        
    except Exception as e:
        print(f"Error invoking agent: {e}")
        return {"error": str(e)}
    

if __name__ == "__main__":
    # Load queries from the input file
    input_file = Path(__file__).resolve().parents[0] / "eval_data.jsonl"
    print(f"Input file: {input_file}")

    batch_evaluation(input_file)
    
    print("Batch evaluation completed successfully.")

    print("☁️  Deploying to Azure AI Foundry...")
    
    result = evaluate(
        data=input_file,
        evaluators=quality_and_safety_evaluators,
        evaluator_config=evaluator_configs,
        azure_ai_project=azure_ai_project,
        output_path= Path(__file__).resolve().parents[0] / "./cloud_results_enhanced.json"
    )
