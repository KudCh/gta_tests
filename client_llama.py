import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_chat(model_name: str, server_path: str):
    # LM Studio or any OpenAI-compatible server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Start MCP tool server
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            print("Tools available:", [t.name for t in tools_response.tools])

            history = []

            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    break

                history.append({"role": "user", "content": user_input})

                # Stream assistant response
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=history,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                        for tool in tools_response.tools
                    ],
                    stream=True,
                )

                print("Bot: ", end="", flush=True)
                full_message = {"role": "assistant", "content": ""}
                tool_calls = []

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Handle text tokens
                    if delta.get("content"):
                        print(delta["content"], end="", flush=True)
                        full_message["content"] += delta["content"]

                    # Handle tool calls (function_call in legacy API)
                    if "tool_calls" in delta:
                        tool_calls.extend(delta["tool_calls"])

                print("")  # newline after streamed output
                history.append(full_message)

                # If a tool was requested
                if tool_calls:
                    for call in tool_calls:
                        tool_name = call.function.name
                        tool_args = json.loads(call.function.arguments)

                        result = await session.call_tool(tool_name, tool_args)

                        # Add tool result to history
                        history.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": str(result),
                        })

                        # Follow up with model (non-streaming for simplicity)
                        followup = client.chat.completions.create(
                            model=model_name,
                            messages=history,
                        )
                        answer = followup.choices[0].message.content
                        history.append({"role": "assistant", "content": answer})
                        print("Bot (final):", answer)


if __name__ == "__main__":
    asyncio.run(run_chat("tinyllama-1.1b-chat-v1.0", "weather_server.py"))
