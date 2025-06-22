# Agentic RAG with MCP Web Search

An intelligent RAG system that combines document search with web search using LLM-based tool calling. The system automatically decides when to search the web based on document context quality and question requirements.

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

3. **Use the system**:
   ```python
   from agentic_rag_mcp import RAGOrchestrator
   
   rag = RAGOrchestrator()
   rag.ingest('document.pdf')
   
   # Document-based query
   answer = rag.query("What is the main topic?")
   
   # May trigger web search
   answer = rag.query("What are the latest developments?")
   ```

## How It Works

1. **Document Search**: Retrieves relevant chunks from ingested PDFs
2. **Context Evaluation**: LLM receives document context via system prompt
3. **Tool Decision**: LLM decides whether to call `search_web` tool
4. **Web Search**: If needed, searches web via MCP server
5. **Answer Generation**: Combines document and web contexts

## Configuration

```python
rag = RAGOrchestrator(
    n_candidates=3,        # Number of retrieval candidates
    k=5,                   # Chunks per candidate
    mcp_endpoint="http://localhost:8000",
    mcp_apikey="your-key"
)
```

## Architecture

- **RAGOrchestrator**: Main coordinator
- **WebSearchAgent**: MCP server communication
- **RetrievalAgent**: Multi-candidate document search
- **RankingAgent**: LLM-based answer selection
- **Tool Calling**: LLM decides when to search web

## Requirements

- Python 3.8+
- OpenAI API key
- MCP server (optional, for web search)

## License

MIT License