import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

async def run_chat(model_name: str):

    # Connect to the official OpenAI API
    client = OpenAI()  

    history = []

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
                    "server_url": "https://923dbd45fa7f.ngrok-free.app/sse",
                    "require_approval": "never",
                },
            ],
        )

        # The model may return multiple items (messages, tool calls, etc.)
        for output in response.output:
            print(output) 
            if output.type == "message":
                msg = output
                if msg.role == "assistant":
                    # Check if assistant requested a tool
                    if hasattr(msg, "tool_calls"):
                        for call in msg.tool_calls:
                            tool_name = call.function.name
                            tool_args = json.loads(call.function.arguments)

                            result = await session.call_tool(tool_name, tool_args)

                            # Add tool result to history
                            history.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": str(result),
                            })

                            # Ask again with tool result
                            followup = client.responses.create(
                                model=model_name,
                                input=history,
                            )
                            answer = followup.output[0].content[0].text
                            history.append({"role": "assistant", "content": answer})
                            print("Bot:", answer)

                    else:
                        # Plain assistant reply
                        answer = msg.content[0].text
                        history.append({"role": "assistant", "content": answer})
                        print("Bot:", answer)


if __name__ == "__main__":
    asyncio.run(run_chat("gpt-4.1-mini"))
