import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import sys
import base64

load_dotenv()  

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
async def run_chat(model_name: str, server_url: str, test_index: str = "0"):

    # Connect to the official OpenAI API
    client = OpenAI()  

    history = [
        {
            "role": "system",
            "content": (
                "You are an assistant that always reasons in the ReAct style.\n"
                # "For every user query:\n"
                # "- Start with Thought: (your reasoning)\n"
                # "- If a tool is needed, output Action: (tool name and arguments)\n"
                # "- When tool results are available, output Observation: (tool output)\n"
                # "- Finally, provide the final answer.\n"
                # "Do not skip Thought/Action/Observation, even if no tool is required."
                """Instructions:
                1. Analyze the query, previous reasoning steps, and observations.
                2. Decide on the next action: use a tool or provide a final answer.
                3. Respond in the following JSON format:

                If you need to use a tool:
                {{
                    "thought": "Your detailed reasoning about what to do next",
                    "action": {{
                        "name": "Tool name (wikipedia, google, or none)",
                        "reason": "Explanation of why you chose this tool",
                        "input": "Specific input for the tool, if different from the original query"
                    }}
                }}

                If you have enough information to answer the query:
                {{
                    "thought": "Your final reasoning process",
                    "answer": "Your comprehensive answer to the query"
                }}"""
            ),
        }
    ]

    content = []
    with open("gta_dataset/dataset.json", "r") as f:
        
        data = json.load(f)

        # load test query
        test_query = data[test_index]["dialogs"][0]["content"] 
        print("Test query:", test_query)
        content.append({ "type": "input_text", "text": test_query})
        
        # load test images 
        for image in data[test_index]["files"]:
            if image["url"]: 
                content.append({ "type": "input_image", "image_url": image["url"]})
            else: 
                base64_image = encode_image("gta_dataset" + image["path"])
                content.append({ "type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"})

    # Add user input to history
    history.append({"role": "user", "content": json.dumps(content)})

    # Ask the model
    response = client.responses.create(
        model=model_name,
        input=history,
        tools=[
            {
                "type": "mcp",
                "server_label": "gta-tools",
                "server_description": "Server with tools from AgentLego used in the GTA paper",
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
