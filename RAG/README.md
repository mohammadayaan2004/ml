# Retrieval-Augmented Generation (RAG) Chatbot

A document-aware LLM assistant that combines a vector database with a language model to provide grounded responses with fewer hallucinations and citeable outputs.

## Features

- Document indexing with sentence-transformer embeddings
- FAISS vector database for fast similarity search
- Transformer-based language model for response generation
- Web interface using Streamlit
- Command-line interface for quick testing
- Interactive chat experience

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Recommended)

Run the Streamlit web interface:
```bash
streamlit run web_interface.py
```

On Windows, you can double-click the `start_chatbot.bat` file or run:
```cmd
start_chatbot.bat
```

Or use the Python launcher script:
```bash
python launch_app.py
```

Then:
1. Add documents using the sidebar (upload a text file or add manually)
2. Ask questions in the chat interface
3. View retrieved documents and generated responses

### Command-Line Interface

#### Interactive Mode
```bash
python cli_interface.py --interactive
```

#### Single Query Mode
```bash
python cli_interface.py "What is photosynthesis?"
```

#### With Documents
```bash
python cli_interface.py --docs "Document 1" "Document 2" --interactive
```

## How It Works

1. **Document Indexing**: Documents are embedded using sentence-transformers and stored in a FAISS vector database
2. **Retrieval**: When a query is received, it's embedded and similar documents are retrieved from the database
3. **Generation**: The retrieved documents are used as context for a language model to generate a grounded response

## Customization

- Change the embedding model by modifying the `embedding_model` parameter in [rag_chatbot.py](rag_chatbot.py)
- Change the generative model by modifying the `model_name` parameter in [rag_chatbot.py](rag_chatbot.py)
- Adjust the number of retrieved documents by changing the `k` parameter in the `retrieve_documents` method

## Sample Documents

The repository includes [sample_documents.txt](sample_documents.txt) which you can use to test the chatbot.

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.