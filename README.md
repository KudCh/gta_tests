# GTA tools server & client 
Implementation of an MCP server with tools from the [GTA benchmark](https://github.com/open-compass/GTA/tree/main) and a simple OpenAI chatbot implemented with Response API 

## Files
**agentlego**: GTA adaptation of the Agentlego tools library

**gta_dataset**: dataset with GTA benchmark tasks and images 

**gta_server.py**: FastMCP server with GTA bencmark tool wrappers 

**gta_client.py**: ReAct style chatbot with an MCP client that connects to the GTA tools server 

## Installation 
1. Install Agengtlego dependencies 

```bash
conda create -n agentlego python=3.11.9
conda activate agentlego
cd agentlego
pip install -r requirements_all.txt
pip install agentlego
pip install -e .
mim install mmengine
mim install mmcv==2.1.0
```

2. Install ngrok 

3. To use the client, you need to an OpenAI API key strored in a .env file

```python
OPENAI_API_KEY='your API key'
```
4. To use the GoogleSearch and MathOCR tools, you should first get the Serper API key from https://serper.dev, and the Mathpix API key from https://mathpix.com/. Then add these to .env or export them as environment variables

```python
SERPER_API_KEY='your_serper_key_for_google_search_tool'
MATHPIX_APP_ID='your_mathpix_key_for_mathocr_tool'
MATHPIX_APP_KEY='your_mathpix_key_for_mathocr_tool'
```

## Run the program:

1. Run tools server  
```
uv run gta_server.py
```
2. Redirect the server to an external address with ngrok 
```
ngrok http 8000
```
3. Run the chatbot file with the URL generated from ngrok and the index of the GTA task you wish to run 
```
uv run gta_client.py <ngrok_url> <task_id>
``` 