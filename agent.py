from databricks.sdk import WorkspaceClient

import mlflow
#from mlflow.tracing.destination import MlflowExperiment
from mlflow import tracing
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import json
import time
from datetime import timedelta
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from openai import AsyncOpenAI
import requests 
from requests.exceptions import RequestException
from typing import List, Generator, Any, Optional, Dict
import yaml
import os
import re 
from uuid import uuid4


from src.functions import get_weather, call_chat_model, create_tool_calls_output
from src.config import get_raw_configs
from src.general_functions import get_workspace_client
from src.genie_functions import run_genie
#### Set up agent monitoring

mlflow.openai.autolog(log_traces=False)

experiment_name = "/dbx_agent_demo"
mlflow.set_experiment(experiment_name)

# Get the experiment details
experiment = mlflow.get_experiment_by_name(experiment_name)
tracing.set_destination(
  tracing.destination.Databricks(experiment_id=experiment.experiment_id)
)

class WeatherAgent(ChatAgent):
    def __init__(self):

        self.get_configs()  # Automatically call configs   
        self.w_system = get_workspace_client('system')
        self.openai_client = self.w_system.serving_endpoints.get_open_ai_client()

    def get_configs(self):
        config_data = get_raw_configs()
        self.system_prompt = config_data['system_prompt']
        self.model_name = config_data['model_name']
        self.tools = [value for key, value in config_data['tools'].items()]
        self.function_mapping = {tool['function']['name']: globals().get(tool['function']['name']) for tool in self.tools}

    @mlflow.trace(name="stringify_tool_call", span_type=SpanType.CHAIN)    
    def stringify_tool_call(self, response: object) -> dict:
        """
        Extracts and formats information from a tool call response.

        Parameters:
            response (object): The response object containing choices, messages, and tool call details.

        Returns:
            dict: A dictionary with the role, content, and tool call details from the response.
        """

        try:
            # Extract message details
            value = response.choices[0].message

            # Construct payload
            payload = {
                "role": value.role,
                "content": value.content,
                "name": None,  # No name provided, use None
                "id": response.id,  # New id with 'run-' prefix
                "tool_calls": create_tool_calls_output(value),
                "tool_call_id": None,  # No tool_call_id in the input, set to None
                "attachments": None  # No attachments in the input, set to None
            } 

            return ChatAgentMessage(**payload) 
        
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Invalid response format: {e}")
         

    @mlflow.trace(name="process_tool_calls", span_type=SpanType.CHAIN)        
    def process_tool_calls(self, response: object) -> list:
        """
        Processes a list of tool calls and maps them to appropriate functions for execution.

        Parameters:
            tool_calls (object): A list of tool call objects containing function details.

        Returns:
            list: A list of payload dictionaries representing the processed tool calls.
        """
        # Fetching the assistant message
        new_messages = []
        new_messages.append(self.stringify_tool_call(response))

        # Start processing tool call one by one
        tool_calls = response.choices[0].message.tool_calls

        for tool_call in tool_calls:
            try:
                # Extract function name and arguments
                function_name = tool_call.function.name
                if function_name not in self.function_mapping:
                    raise ValueError(f"Unknown function name: {function_name}")
                
                print(f"Function called {function_name} has been triggered")
                # Get the mapped function
                function = self.function_mapping[function_name]

                # Parse the function arguments
                function_arguments = json.loads(tool_call.function.arguments)
                # Execute the function
                function_output = function(**function_arguments)

                # Add validator that function_output is ALWAYS a string
                if not isinstance(function_output, str):
                    raise ValueError(f"Function '{function_name}' did not return a string.")

                # Create the payload
                payload = {
                    "role": "tool",
                    "content": function_output,  # Call the mapped function
                    "name": function_name,
                    "tool_call_id": tool_call.id,
                    "id": str(uuid4())
                }

                # Append the processed payload
                new_messages.append(ChatAgentMessage(**payload))
            except Exception as e:
                raise RuntimeError(f"Error processing tool call {tool_call.id}: {e}")
        
        return new_messages


    def agent_tool_calling(self, messages):

        temperature = 0.3
        max_tokens= 700
        max_node_count = 6
        node_count = 1

        try:
            result = call_chat_model(self.openai_client, self.model_name, messages, temperature, max_tokens, tools = self.tools)
            if result.choices[0].finish_reason == 'length':
                yield "You are running out of tokens. Please reduce the number of tokens or increase the node count."

            if result.choices[0].finish_reason == 'stop':
                yield ChatAgentMessage(**result.choices[0].message.to_dict(), id=result.id)

            # Activating the Agent
            while result.choices[0].finish_reason != 'stop' and node_count <= max_node_count:

                # Processing tool_calls
                for temp_message in self.process_tool_calls(result):
                    print(temp_message)
                    print("#LOL")
                    messages.append(temp_message)
                    yield temp_message

                # adding one full node count
                node_count += 1

                # Calling chat model again
                result = call_chat_model(self.openai_client, self.model_name, messages, temperature, max_tokens, tools = self.tools)
                yield ChatAgentMessage(**result.choices[0].message.to_dict(), id=result.id)
        
        except Exception as e:
            print(f'Error occurred: {e}')

    @mlflow.trace(name="Agent Executor", span_type='AGENT')
    def predict(self, 
                messages: List[ChatAgentMessage],
                context: Optional[ChatContext] = None,
                custom_inputs: Optional[dict[str, Any]] = None,
                ) -> ChatAgentResponse:  

        message_state = [
            ChatAgentMessage(role="system", content=self.system_prompt)
            ] + messages

        response_messages = [
            chunk.delta
            for chunk in self.predict_stream(message_state, context, custom_inputs)
        ]
        return ChatAgentResponse(messages=response_messages)


    @mlflow.trace(name="Agent Executor", span_type='AGENT')
    def predict_stream(  
        self,  
        messages,  # Should be List[ChatAgentMessage]  
        context=None,  
        custom_inputs=None,  
    ):  
        message_state = [
            ChatAgentMessage(role="system", content=self.system_prompt)
            ] + messages
    
        #stream = call_chat_model(self.openai_client, self.model_name, message_state, stream=True)
        for message in self.agent_tool_calling(messages=message_state):
            yield ChatAgentChunk(delta=message)

agent = WeatherAgent()
mlflow.models.set_model(agent)
