import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import sys

load_dotenv()  # Load environment variables from .env file

async def run_chat(model_name: str, server_url: str = "http://localhost:8000"):

    # Connect to the official OpenAI API
    client = OpenAI()  

    history = [
        {
            "role": "system",
            "content": (
                "You are an assistant that always reasons in the ReAct style.\n"
                "For every user query:\n"
                "- Start with Thought: (your reasoning)\n"
                "- If a tool is needed, output Action: (tool name and arguments)\n"
                "- When tool results are available, output Observation: (tool output)\n"
                "- Finally, provide the final answer.\n"
                "Do not skip Thought/Action/Observation, even if no tool is required."
            ),
        }
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        # Add user input to history
        history.append({"role": "user", "content": user_input})

        # Ask the model
        response = client.responses.create(
            model=model_name,
            input=history,
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_description": "Weather MCP server with forecast and alerts",
                    "server_url": server_url + "/sse",
                    "require_approval": "never",
                },
            ],
        )

        for output in response.output:
            if output.type == "message":
                role = output.role
                parts = []
                for c in output.content:
                    if c.type == "output_text":
                        parts.append(c.text)
                content = "".join(parts)

                if role == "assistant":
                    print(f"Bot:\n{content}")  # full ReAct trace
                    history.append({"role": "assistant", "content": content})

                elif role == "tool":
                    print(f"Observation: {content}")
                    history.append({"role": "tool", "content": content})


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python openai_mcp_agent.py [server_url]")
        print("Example: python openai_mcp_agent.py https://cfe484f994aa.ngrok-free.app")
        sys.exit(1)
    
    asyncio.run(run_chat("gpt-4.1-mini", sys.argv[1]))
