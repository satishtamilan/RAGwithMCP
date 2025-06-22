# Agentic RAG with MCP Web Search

An intelligent RAG system that combines document search with web search using LLM-based tool calling. The system automatically decides when to search the web based on document context quality and question requirements.

This project was built as part of the Blog [Hands on — Agentic RAG (3/3) — Agentic Integrated with MCP Server](https://abvijaykumar.medium.com/hands-on-agentic-rag-3-3-agentic-integrated-with-mcp-server-055e1d601608)

You can read the other 2 parts here
* [Hands on — Agentic RAG (1/3)](https://abvijaykumar.medium.com/hands-on-agentic-rag-1-2-cdf375ad7e7a)
* [Hands on — Agentic RAG (2/3) — Agentic ReRanking RAG](https://abvijaykumar.medium.com/hands-on-agentic-rag-2-3-agentic-reranking-rag-773b04cf4cdd)



## Features

- **Smart Decision Making**: LLM evaluates document context and decides when web search is needed
- **MCP Integration**: Web search via Model Context Protocol servers
- **Multi-Candidate Retrieval**: Parallel retrieval with LLM-based ranking
- **Document-First Approach**: Prioritizes local documents, uses web as intelligent fallback

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export MCP_ENDPOINT="http://localhost:8000"  # Optional
   ```



## How It Works

1. **Document Search**: Retrieves relevant chunks from ingested PDFs
2. **Context Evaluation**: LLM receives document context via system prompt
3. **Tool Decision**: LLM decides whether to call `search_web` tool
4. **Web Search**: If needed, searches web via MCP server
5. **Answer Generation**: Combines document and web contexts


## Architecture

- **RAGOrchestrator**: Main coordinator
- **WebSearchAgent**: MCP server communication
- **RetrievalAgent**: Multi-candidate document search
- **RankingAgent**: LLM-based answer selection
- **Tool Calling**: LLM decides when to search web

Read the blog for more details

## Requirements

- Python 3.8+
- OpenAI API key
- MCP server (optional, for web search)

## License

MIT License