# Agentic RAG with MCP Web Search

An intelligent RAG system that combines document search with web search using LLM-based tool calling. The system automatically decides when to search the web based on document context quality and question requirements.

This project was built as part of the Blog [Hands on — Agentic RAG (3/3) — Agentic Integrated with MCP Server](https://abvijaykumar.medium.com/hands-on-agentic-rag-3-3-agentic-integrated-with-mcp-server-055e1d601608)

You can read the other 2 parts here
* [Hands on — Agentic RAG (1/3)](https://abvijaykumar.medium.com/hands-on-agentic-rag-1-2-cdf375ad7e7a)
* [Hands on — Agentic RAG (2/3) — Agentic ReRanking RAG](https://abvijaykumar.medium.com/hands-on-agentic-rag-2-3-agentic-reranking-rag-773b04cf4cdd)



## Features

- **📄 Multiple PDF Support**: Upload and process multiple PDF documents simultaneously
- **🤖 Smart Decision Making**: LLM evaluates document context and decides when web search is needed
- **🌐 MCP Integration**: Web search via Model Context Protocol servers
- **🔍 Multi-Candidate Retrieval**: Parallel retrieval with LLM-based ranking
- **📚 Document-First Approach**: Prioritizes local documents, uses web as intelligent fallback
- **🎯 Interactive UI**: Streamlit-based web interface with document management

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd agentic-rag-mcp
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r mcp_web_search/requirements.txt
   pip install fastapi  # Required for MCP server
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Required for the main RAG application
   OPENAI_API_KEY=your-openai-api-key-here
   
   # Optional - for web search functionality via MCP server
   SERPAPI_KEY=your-serpapi-key-here
   
   # Optional - MCP server endpoint (defaults to http://localhost:8000)
   MCP_ENDPOINT=http://localhost:8000
   ```

5. **Start the MCP server** (optional, for web search):
   ```bash
   source .env
   cd mcp_web_search
   python web_search_server.py
   ```

6. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

7. **Open your browser** and go to `http://localhost:8501`



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

## Usage

### Uploading Documents
1. Click "Browse files" to select one or more PDF documents
2. Review the selected files and their sizes
3. Click "🔄 Ingest All PDFs" to process and index the documents
4. Wait for the processing to complete

### Asking Questions
1. Once documents are loaded, enter your question in the text input
2. The system will:
   - Search through your uploaded documents
   - Decide if web search is needed for additional context
   - Provide a comprehensive answer combining both sources

### Managing Documents
- View currently loaded documents in the sidebar
- Clear all documents to start fresh with new files
- Upload additional documents to expand your knowledge base

## Testing MCP Integration

To verify the MCP server is working:

```bash
# Test basic connectivity
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize"}'

# Test web search functionality
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"search","params":{"query":"AI news","max_results":3}}'
```

## Project Structure

```
agentic-rag-mcp/
├── app.py                      # Streamlit web application
├── agentic_rag_mcp.py         # Main RAG orchestrator and agents
├── requirements.txt           # Python dependencies
├── mcp_web_search/           # MCP server for web search
│   ├── web_search_server.py  # FastAPI-based MCP server
│   ├── client.py             # MCP client utilities
│   └── requirements.txt      # MCP server dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- OpenAI API key (required)
- SerpAPI key (optional, for web search)
- FastAPI and MCP dependencies

## Troubleshooting

### Common Issues

1. **"No module named 'fastapi'"**: Install FastAPI with `pip install fastapi`
2. **MCP server not responding**: Check if port 8000 is available and environment variables are set
3. **OpenAI API errors**: Verify your API key is correct and has sufficient credits
4. **PDF processing errors**: Ensure uploaded files are valid PDF documents

### Environment Variable Issues

If environment variables aren't working:
1. Ensure the `.env` file is in the project root
2. Use `source .env` before starting the MCP server
3. Check that variable names match exactly (case-sensitive)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License