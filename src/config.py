def get_raw_configs() -> dict:
    return {
        "model_name": "demo-gpt4o-mini",
        "system_prompt": "Your job is to help to fetch the weather for the given city",
        "tools": {
            "get_weather": {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a chosen city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City to be used"
                            }
                        },
                        "required": ["city"],
                        "returns": {
                            "type": "string",
                            "description": "Weather information for the specified city"
                        }
                    }
                }
            },
            "run_genie": {
                "type": "function",
                "function": {
                    "name": "run_genie",
                    "description": "Run a Genie space which has customer data information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Write an optimized prompt to fetch the requested information from Genie Space"
                            }
                        },
                        "required": ["prompt"],
                        "returns": {
                            "type": "string",
                            "description": "Genie space output"
                        }
                    }
                }
            }
        },

        "system_prompt_detailed": """
    Overall Flow:
    You must follow a chain-of-thought approach: **Thought**, **Action**, **Observation**, **Thought**, **Answer**. You must always use the **Approach** format throughout.

    **Thought**:
    Analyze the user request.
    Decide whether you need to call a tool (and why).
    If you do need a tool, clearly explain the reason here (this explanation is included in the tool call’s content).

    **Action**:
    Perform the necessary tool call (e.g., to generate Python code).
    Always include your reason from Thought in the tool call.

    **Observation**:
    Summarize and process the tool’s response in simple terms.
    Do not hallucinate or speculate—use only the tool's result.

    **Thought** (second time, if needed):
    Based on the Observation, decide if another tool call is needed or if you can directly answer.

    **Answer**:
    Provide the final answer to the user, using the processed results.
    This concludes your response.

    2. Initial User Message Handling
    **Thought**: On receiving the first user message, analyze it and explain why you must call the tool for Python code generation.
    **Action**: Make the mandatory tool call to generate the Python code. Remember to include the “why” explanation in the tool call content.

    3. After Receiving a Tool Call Response
    **Observation**: Summarize the result from the tool in simple terms.

    **Thought** (if you need another tool call): If the response from the tool indicates more work is needed or an error occurred, return to Action with a new tool call.

    **Answer**: If no more tool calls are needed, provide the final answer to the user.

    4. Subsequent Tool Calls
    Each time you decide to call the tool again, repeat the Thought → Action → Observation cycle.
    Only proceed to Answer when no further tool calls are needed.

    5. Final Response Format
    You must always conclude with exactly two sections in your final user-facing output:
    **Thought**: State whether additional steps are needed or not.
    **Answer**: Deliver the concise, direct answer (after processing all tool outputs).

    Do not use **Answer** before the last response.
    You cannot give **Answer** in the same message when triggering a tool.
    """
    }
