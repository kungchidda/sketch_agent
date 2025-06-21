#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "anthropic>=0.45.0",
#   "openai>=1.24.0",
# ]
# ///
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union

import anthropic
import openai
import json



# In[ ]:


def main():
    try:
        print("\n=== LLM Agent Loop with Claude and Bash Tool ===\n")
        print("Type 'exit' to end the conversation.\n")
        provider = os.getenv("MODEL_PROVIDER", "anthropic").lower()
        if provider == "openai":
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            loop(LLMOpenAI(model_name))
        else:
            loop(LLM("claude-3-7-sonnet-latest"))
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")



# In[ ]:


def loop(llm):
    provider = getattr(llm, "provider", "anthropic")
    msg = user_input()            # first user msg
    while True:
        # ① assistant responds
        output, tool_calls = llm(msg)
        if output:
            print("Agent: ", output)

        # ② if assistant called tools → run them & feed results back
        if tool_calls:
            tool_results = [handle_tool_call(tc, provider) for tc in tool_calls]
            # Immediately send tool results back to the LLM
            msg = tool_results
            continue              # repeat loop without new user input
        # ③ no tool calls → ask user again
        msg = user_input()




# In[ ]:


bash_tool = {
    "name": "bash",
    "description": "Execute bash commands and return the output",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            }
        },
        "required": ["command"]
    }
}



# In[ ]:


# Function to execute bash commands
def execute_bash(command):
    """Execute a bash command and return a formatted string with the results."""
    # If we have a timeout exception, we'll return an error message instead
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=10
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
    except Exception as e:
        return f"Error executing command: {str(e)}"



# In[ ]:


def user_input():
    x = input("You: ")
    if x.lower() in ["exit", "quit"]:
        print("\nExiting agent loop. Goodbye!")
        raise SystemExit(0)
    return [{"type": "text", "text": x}]



# In[ ]:


class LLM:
    def __init__(self, model):
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.provider = "anthropic"
        self.messages = []
        self.system_prompt = """You are a helpful AI assistant with access to bash commands.
        You can help the user by executing commands and interpreting the results.
        Be careful with destructive commands and always explain what you're doing.
        You have access to the bash tool which allows you to run shell commands."""
        self.tools = [bash_tool]

    def __call__(self, content):
        self.messages.append({"role": "user", "content": content})
        self.messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20_000,
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tools
        )
        del self.messages[-1]["content"][-1]["cache_control"]
        assistant_response = {"role": "assistant", "content": []}
        tool_calls = []
        output_text = ""

        for content in response.content:
            if content.type == "text":
                text_content = content.text
                output_text += text_content
                assistant_response["content"].append({"type": "text", "text": text_content})
            elif content.type == "tool_use":
                assistant_response["content"].append(content)
                tool_calls.append({
                    "id": content.id,
                    "name": content.name,
                    "input": content.input
                })

        self.messages.append(assistant_response)
        return output_text, tool_calls


class LLMOpenAI:
    """
    OpenAI‑based drop‑in replacement for the Anthropic‑backed LLM.
    It expects an environment variable OPENAI_API_KEY and supports the same
    bash tool interface.
    """
    def __init__(self, model: str):
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not found.")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.provider = "openai"
        self.messages = []
        self.system_prompt = """You are a helpful AI assistant with access to bash commands.
You can help the user by executing commands and interpreting the results.
Be careful with destructive commands and always explain what you're doing.
You have access to the bash tool which allows you to run shell commands."""
        # OpenAI requires each tool to be wrapped with {"type": "function", "function": ...}
        self.tools = [{ "type": "function", "function": bash_tool }]

    def __call__(self, content):
        # Detect whether `content` is (a) list of {type:'text'} parts
        #      → user input, or (b) list of {role:'tool', ...} messages.
        if content and isinstance(content, list) and "role" in content[0]:
            # tool result messages; just extend history
            self.messages.extend(content)
        else:
            user_text = " ".join(
                part["text"] for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            self.messages.append({"role": "user", "content": user_text})

        # Call OpenAI Chat Completions with tool support
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
            tools=self.tools,
            temperature=0.7
        )

        assistant_msg = response.choices[0].message
        output_text = assistant_msg.content or ""
        tool_calls = []

        # Parse tool calls (if any)
        if getattr(assistant_msg, "tool_calls", None):
            for tc in assistant_msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments or "{}"),
                    }
                )

        # Keep full assistant message history for context
        self.messages.append(assistant_msg.to_dict())
        return output_text, tool_calls



# In[ ]:


def handle_tool_call(tool_call, provider="anthropic"):
    if tool_call["name"] != "bash":
        raise Exception(f"Unsupported tool: {tool_call['name']}")

    command = tool_call["input"]["command"]
    print(f"Executing bash command: {command}")
    output_text = execute_bash(command)
    print(f"Bash output:\n{output_text}")

    if provider == "openai":
        # OpenAI expects a role='tool' message referencing the call id
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": output_text
        }
    else:  # Anthropic
        return {
            "type": "tool_result",
            "tool_use_id": tool_call["id"],
            "content": [ { "type": "text", "text": output_text } ]
        }



# In[ ]:


if __name__ == "__main__":
    main()
