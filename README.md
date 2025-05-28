# Simple RAG System

A simplified Retrieval-Augmented Generation (RAG) system that allows you to query your own documents using a local language model.

## Features

- **Simple Document Processing**: Upload your PDF, DOCX, and TXT files to process and index
- **Vector Database**: Uses ChromaDB for efficient vector storage and retrieval
- **Language Model Integration**: Connects with Ollama to run open-source LLMs locally
- **Line Reference Tracking**: Shows approximate line numbers for each source passage
- **Terminal Interface**: Clean command-line output format for easy integration

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) installed and running

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/simple-rag-system.git
cd simple-rag-system
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

3. Install and start Ollama:
   - Visit [Ollama.com](https://ollama.com) for installation instructions
   - Start the Ollama service: `ollama serve`
   - Pull the model: `ollama pull llama3.2:3b`

## Usage

### 1. Add Your Documents

Place your documents in the `data` directory:
```bash
mkdir -p data
# Copy your PDF, DOCX, or TXT files to the data directory
```

### 2. Process Documents

Process and index your documents:
```bash
python3 process_documents.py
```

### 3. Query Your Documents

Query directly from the command line:
```bash
python3 query_data.py "Your question about the documents?"
```

Optional parameters:
- `--k`: Number of chunks to retrieve (default: 5)
- `--model`: Ollama model to use (default: "llama3.2:3b")
- `--format`: Output format - "text" or "json" (default: "text")

## How It Works

1. **Document Processing**: Documents are loaded, split into chunks, and converted to embeddings
2. **Storage**: The embeddings and document chunks are stored in a ChromaDB vector database
3. **Retrieval**: When you ask a question, the system finds the most relevant document chunks
4. **Generation**: The LLM uses the retrieved chunks to generate a contextually accurate answer

## Project Structure

- `process_documents.py`: Document loading, splitting, and database population
- `query_data.py`: Document retrieval and query answering
- `get_embedding_function.py`: Embedding model configuration
- `data/`: Directory for your documents
- `chroma/`: Vector database storage (created automatically)

## Customization

- Change the embedding model in `get_embedding_function.py`
- Adjust chunk size and overlap in `process_documents.py`
- Modify the prompt template in `query_data.py`

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Missing Models**: Pull required models with `ollama pull MODEL_NAME`
- **Empty Results**: Check that documents were properly processed by examining the console output

## License

MIT
