from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings

# load OpenAI API key
load_dotenv(find_dotenv())

prompt = (

    "You are a ReAct style agent. "
    "You will get queries that consist of a task that you need to perform "
    "and a list of images that you will need to perform the task."

    "You will be given a list of tools that you must use to perform the task. " 
    "For each reasoning step, produce a result in the following format:" \
    
    "Thought: ..." \
    "Action: ..." \
    "Action Output: ... " \

    "When you are done, write: " \
    "Final Answer: ..."

)
 

MCP_SERVER_URL = "http://localhost:8000/sse"

async def run_task(model_name, query, files, tools_server_url=MCP_SERVER_URL):

    async with MCPServerSse(
        name = "GTA Tools Server", 
        params = {
            "url": tools_server_url
        }, 
        cache_tools_list=True
    ) as server:
        agent = Agent(
            name="GTA benchmark agent",
            mcp_servers = [server],
            model_settings=ModelSettings(tool_choice="required"),
            instructions=prompt, 
            model=model_name
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        for image_path in files:
            messages.append({
            "role": "user",
            "content": [{"type": "text", "text": image_path}]
            })

        result = await Runner.run(agent, messages) # type: ignore

    return result    

def main():
    import json
    import argparse
    TASKS_FILE = "gta_dataset/dataset.json"
    RESULTS_FILE = "results.json"

    parser = argparse.ArgumentParser(prog='OpenAI MCP client')

    parser.add_argument("--start", type=int, required=True, help="Task index to start testing from")
    parser.add_argument("--end", type=int, required=True, help="Task index to finish testing at (inclusive)")
    parser.add_argument("--tools_server_url", type=str, required=True, help="Server link to connect to (e.g. http://localhost:8000)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM to be tested")
    args = parser.parse_args()

    # Print what will run
    print(f"Running tasks {args.start}-{args.end} on model {args.model_name}")

    with open(TASKS_FILE) as f:
        tasks = json.load(f)
    
    results = []

    # Loop over the task range
    for task_id in range(args.start, args.end + 1):
        print(f"\n>>> Running task {task_id}")

        task = tasks[task_id]["dialogs"][0]["content"]
        files = [file["path"] for file in tasks[task_id]["files"]]

        output = run_task(args.model_name, task, files, args.tools_server_url)
        results.append({
                        "task_id": task_id, 
                        "output": output
                    })

    with open(RESULTS_FILE, "a") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()