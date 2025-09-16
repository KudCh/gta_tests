import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_react_agent(model_name: str, server_path: str, max_steps: int = 5):
    """
    Fully async ReAct agent loop using OpenAI Responses API and MCP tool server.
    """

    # Initialize OpenAI client
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Start MCP server
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools from MCP
            tools_response = await session.list_tools()
            available_tools = {t.name: t for t in tools_response.tools}
            print("Tools available:", list(available_tools.keys()))

            # Initialize conversation history
            history = []

            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    break

                history.append({"role": "user", "content": user_input})

                for step in range(max_steps):
                    # Call OpenAI Responses API
                    response = client.responses.create(
                        model=model_name,
                        input=history,
                        tools=[
                            {
                                "type": "function",
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            }
                            for tool in tools_response.tools
                        ]
                    )

                    ai_output = response.output_text
                    print(f"\nStep {step+1} - AI Output:\n{ai_output}")

                    # Check for tool calls
                    tool_calls = getattr(response, "tool_calls", [])
                    if tool_calls:
                        for call in tool_calls:
                            tool_name = call.function.name
                            tool_args = json.loads(call.function.arguments)
                            if tool_name in available_tools:
                                result = await session.call_tool(tool_name, tool_args)
                            else:
                                result = f"Unknown tool: {tool_name}"

                            print(f"Tool Call: {tool_name}({tool_args}) -> Observation: {result}")

                            # Add observation to history
                            history.append({
                                "role": "assistant",
                                "name": tool_name,
                                "content": str(result),
                            })
                    else:
                        # No more tool calls, treat AI output as final answer
                        history.append({"role": "assistant", "content": ai_output})
                        print("Final Answer:", ai_output)
                        break  # exit multi-step loop

if __name__ == "__main__":
    asyncio.run(run_react_agent("tinyllama-1.1b-chat-v1.0", "weather_server.py"))
