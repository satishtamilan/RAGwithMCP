# MCP Web Search Server and Client

A simple Model Context Protocol (MCP) web search server and client that enable searching topics via external web search (powered by SerpAPI) and optional LLM-based decision-making.

## Features

- JSON-RPC 2.0 over HTTP for standard search requests
- Server-Sent Events (SSE) endpoint for streaming search results
- LLM-based client that can decide when to perform web search

## Requirements

- Python 3.8+
- SerpAPI API key (for server)
- OpenAI API key (for LLM-based client, optional)

## Installation

1. Clone the repository or navigate to this directory:
   ```bash
   cd mcp_web_search
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file or set environment variables directly:

```bash
export SERPAPI_KEY="your-serpapi-api-key"
# Optional: Override default server URL for client
export OPENAI_API_KEY="your-openai-api-key"
```

## Running the Search Server

By default, the server listens on `0.0.0.0:8000`.

```bash
# With venv activated and SERPAPI_KEY set
python web_search_server.py
``` 

You should see:
```
[INFO] MCP Search Server is starting...
MCP Search Server HTTP listening at: http://localhost:8000/
```

### Endpoints

- POST `/`
  - JSON-RPC 2.0 endpoint
  - Methods supported:
    - `initialize`
    - `list_tools`
    - `search` (params: `query` or `topic`, and optional `max_results`)
    - `shutdown` (demo only)

- GET `/stream?topic=<topic>&max_results=<n>`
  - SSE endpoint that streams individual results as events

## Using the Python Client

The `client.py` script provides two main functions:

1. **Direct tool call**
   ```python
   from client import SearchClient
   import asyncio

   async def main():
       client = SearchClient(server_url="http://localhost:8000")
       result = await client.call_search_tool("fastapi tutorial", max_results=3)
       print(result)

   asyncio.run(main())
   ```

2. **LLM-based tool calling** (requires OpenAI API key)
   ```bash
   python client.py
   ```
   The script will prompt for queries and use GPT-3.5-turbo to decide if a web search is needed.

## Testing

No tests included by default. You can manually call endpoints or use the client.

## License

MIT License
