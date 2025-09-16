import asyncio
from openai import OpenAI  # assuming you use openai SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_chat(model_name: str, server_path: str):

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    server_params = StdioServerParameters(
        command="python",           
        args = [server_path], 
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()   # ensure session is ready
            tools_response = await session.list_tools()
            print("Tools available:", [t.name for t in tools_response.tools])

            history = []

            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    break

                history.append({"role": "user", "content": user_input})

                response = client.chat.completions.create(
                    model=model_name,
                    messages=history,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            }
                        }
                        for tool in tools_response.tools
                    ]
                )

                choice = response.choices[0]
                msg = choice.message

                if msg.tool_calls:
                    for call in msg.tool_calls:
                        tool_name = call.function.name
                        import json
                        tool_args = json.loads(call.function.arguments)

                        result = await session.call_tool(tool_name, tool_args)

                        # add to history
                        history.append({
                            "role": "assistant",
                            "name": tool_name,
                            "content": str(result),
                        })

                        # Now ask model again with that tool output
                        followup = client.chat.completions.create(
                            model=model_name,
                            messages=history,
                        )
                        answer = followup.choices[0].message.content
                        history.append({"role": "assistant", "content": answer})
                        print("Bot:", answer)
                else:
                    answer = msg.content
                    history.append({"role": "assistant", "content": answer})
                    print("Bot:", answer)


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "tinyllama-1.1b-chat-v1.0"
    server_path = sys.argv[2] if len(sys.argv) > 2 else "weather_server.py"
    asyncio.run(run_chat(model_name, server_path))