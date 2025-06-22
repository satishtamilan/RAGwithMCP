#!/usr/bin/env python3
"""
Simple MCP Server for Topic Search (HTTP Version)
This server provides a search tool that can search for information about a given topic
via JSON-RPC 2.0 over HTTP at a specified host/port. It also supports Server-Sent Events (SSE) for streaming results.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"


@app.on_event("startup")
async def startup_event():
    logger.info("Starting MCP Search Server...")
    host_display = HTTP_HOST if HTTP_HOST != "0.0.0.0" else "localhost"
    print(f"MCP Search Server HTTP listening at: http://{host_display}:{HTTP_PORT}/", flush=True)


@app.post("/")
async def mcp_entrypoint(request: Request):
    """Main MCP JSON-RPC endpoint."""
    body = await request.body()
    try:
        rpc = json.loads(body)
    except Exception:
        response = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}
        logger.info(f"Response: {json.dumps(response, indent=2)}")
        return response

    method = rpc.get("method", "")
    response: Dict[str, Any] = {"jsonrpc": "2.0", "id": rpc.get("id")}

    if method == "initialize":
        response["result"] = {"server": "search-server"}
        logger.info("Initialization received.")
    elif method == "list_tools":
        response["result"] = await list_tools()
        logger.info("Tool list served.")
    elif method == "search":
        params = rpc.get("params", {})
        topic = params.get("query", params.get("topic", ""))
        max_results = params.get("max_results", 5)
        if not topic:
            response["error"] = {"code": -32602, "message": "Missing 'query' or 'topic' parameter."}
            logger.warning("Search call with missing topic/query parameter.")
        else:
            logger.info(f"Received search for topic: {topic}")
            result = await search_for_topic(topic, max_results)
            response["result"] = result
    elif method == "shutdown":
        response["result"] = "Server shutdown initiated (not actually shutting down in this demo)."
        logger.info("Shutdown requested (ignored in HTTP version).")
    else:
        response["error"] = {"code": -32601, "message": f"Method '{method}' not found."}
        logger.warning(f"Unknown method '{method}' requested by client.")

    logger.info(f"Response: {json.dumps(response, indent=2)}")
    return response


@app.get("/stream")
async def mcp_stream(topic: str, max_results: int = 5):
    """
    SSE endpoint for streaming search results.
    Client connects to /stream?topic=...&max_results=...
    """
    generator = search_for_topic_stream(topic, max_results)
    return StreamingResponse(generator, media_type="text/event-stream")


async def list_tools() -> List[dict]:
    return [
        {
            "name": "search_topic",
            "description": "Search for information about a specific topic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to search for"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                },
                "required": ["topic"]
            }
        }
    ]


async def search_for_topic(topic: str, max_results: int = 5) -> str:
    """
    Search for information about a topic using SerpAPI.
    """
    if not SERPAPI_KEY or SERPAPI_KEY == "YOUR_SERPAPI_KEY":
        return "No SerpAPI key found. Set the SERPAPI_KEY environment variable."
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "q": topic,
                "num": max_results
            }
            response = await client.get(SERPAPI_URL, params=params)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get("organic_results", [])
            if not organic_results:
                return f"No specific information found for '{topic}'. Try searching with different keywords."
            results = []
            for i, item in enumerate(organic_results[:max_results]):
                results.append(f"{i+1}. {item.get('title')}\n{item.get('link')}\n{item.get('snippet', '')}\n")
            return "\n".join(results)
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        return f"Failed to search for '{topic}' due to network error"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"An unexpected error occurred while searching for '{topic}'"


async def search_for_topic_stream(topic: str, max_results: int = 5):
    """
    Async generator to stream search results as SSE events.
    """
    if not SERPAPI_KEY or SERPAPI_KEY == "YOUR_SERPAPI_KEY":
        yield f"data: No SerpAPI key found. Set the SERPAPI_KEY environment variable.\n\n"
        return
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "q": topic,
                "num": max_results
            }
            response = await client.get(SERPAPI_URL, params=params)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get("organic_results", [])
            if not organic_results:
                yield f"data: No specific information found for '{topic}'. Try different keywords.\n\n"
            else:
                for i, item in enumerate(organic_results[:max_results]):
                    event_data = {
                        "index": i+1,
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet", "")
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        yield f"data: Failed to search for '{topic}' due to network error\n\n"
    except Exception as e:
        logger.error(f"Unexpected error during streaming: {e}")
        yield f"data: An unexpected error occurred while streaming results for '{topic}'\n\n"


if __name__ == "__main__":
    print("[INFO] MCP Search Server is starting...", flush=True)
    uvicorn.run("web_search_server:app", host=HTTP_HOST, port=HTTP_PORT, log_level="info")
