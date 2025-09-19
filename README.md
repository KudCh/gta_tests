**client_llama.py** : a ReAct agent built with OpenAI Compatibility API by LMStudio based on the (uses tinyllama-1.1b-chat-v1.0 model from LMStudio). The client uses chat.completions API to communicate with the model, as the Responses API is not supported by LMStudio.

**openai_client.py** : a ReAct agent built with the original OpenAI API (uses gpt-4.1-mini model from OpenAI). The client uses Responses API to communicate with the model. The client requires an OpenAI API that can be generated at the [OpenAI platfrom](https://openai.com/api/)

**weather-server.py** : a weather alerts server built following a tutorial at the Model Context Protocol [website](https://modelcontextprotocol.io/docs/develop/build-server) 

**gta_server.py**: MCP server with GTA tools (OCR, ImageDescription, CountGivenObject)

**gta_client.py**: ReAct style chatbot with an MCP client that connects to the GTA MCP server 

Execution:

1. Run weather_server with "uv run weather_server.py"
2. Redirect the server to an external address with ngrok "ngrok http 8000"
3. Run the MCP client with the URL generated from ngrok "uv run openai_mcp_client.py <ngrok_url>" 