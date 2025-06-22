#!/usr/bin/env python3
"""
MCP Web Search Client with LLM-based tool calling
"""

import json
import httpx
import asyncio
import os
from typing import Dict, Any, List

class SearchClient:
    def __init__(self, server_url: str = "http://localhost:8000", openai_api_key: str = None):
        self.server_url = server_url
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
    async def call_search_tool(self, topic: str, max_results: int = 5) -> str:
        """Call the MCP search server"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "search",
            "params": {"query": topic, "max_results": max_results}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.server_url, json=payload)
            result = response.json()
            
            if "error" in result:
                return f"Error: {result['error']['message']}"
            return result.get("result", "No results found")
    
    async def llm_tool_calling(self, query: str) -> str:
        """Use OpenAI LLM to decide if web search is needed and execute accordingly"""
        if not self.openai_api_key:
            return "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, facts, or answers to questions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {"type": "integer", "description": "Max results", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }]
        
        messages = [{"role": "user", "content": query}]
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json=payload
            )
            result = response.json()
            
            message = result["choices"][0]["message"]
            
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                args = json.loads(tool_call["function"]["arguments"])
                print(f"🔍 LLM decided to use web search for: {args['query']}")
                return await self.call_search_tool(args["query"], args.get("max_results", 5))
            else:
                return message["content"]

async def main():
    client = SearchClient()
    
    print("MCP Web Search Client with LLM Tool Calling")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == 'quit':
            break
            
        try:
            result = await client.llm_tool_calling(query)
            print(f"\nResult:\n{result}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())