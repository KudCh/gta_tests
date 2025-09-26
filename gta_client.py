import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import sys
import base64

load_dotenv()  

# -----------------------------
# Helpers

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def create_file(file_path, client):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# -----------------------------
# Main function to run the chat

async def run_chat(model_name: str, server_url: str, test_index: str = "0"):

    # Connect to the official OpenAI API
    client = OpenAI()  

    history = [
        {
            "role": "system",
            "content": (
                "You are an assistant that always reasons in the ReAct style.\n"
                "You have a list of tools available to you to help in answering user queries.\n"

                """For every user tool that you use, please rpovide the following details:

                {{
                    "thought": "Your detailed reasoning about what to do next",
                    "action": {{
                        "name": "Tool name (wikipedia, google, or none)",
                        "reason": "Explanation of why you chose this tool",
                        "input": "Specific input for the tool, if different from the original query"
                    }}
                    "Observation": "Result from the tool, or explnation of the error if the tool failed"
                }}

                Once you have enough information to answer the query:
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
            #file_id = create_file("./gta_dataset/" + image["path"], client)
            #content.append({ "type": "input_image", "file_id": file_id})
            content.append({"type": "input_text", "text": "Image path: " + "./gta_dataset/" + image["path"]})

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
    if len(sys.argv) < 2:
        print("Usage: python openai_mcp_agent.py [server_url] optional:[text index]")
        print("Example: python openai_mcp_agent.py https://cfe484f994aa.ngrok-free.app 1")
        sys.exit(1)
    
    asyncio.run(run_chat("gpt-4.1-mini", sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0"))
