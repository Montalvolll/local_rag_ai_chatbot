# AI-Assistant Bot

AI-Assistant Bot is an AI-powered conversational assistant for querying local data and metadata. It uses OpenAI's GPT models and LangChain + ChromaDB to provide natural language answers based on a knowledge base of local documents information.

## Features

- Loads and indexes knowledge base metadata
- Uses OpenAI embeddings for semantic search
- Conversational interface for natural language queries
- Supports persistent vectorstore for faster repeated queries
- Loads API key from environment variable or `constants.py` for better security
- Supports `.xlsx` and other file types via the `unstructured` package

## How It Works

- Loads data from `knowledge_base.txt` (or the entire `data/` directory)
- Indexes the data using LangChain and OpenAIEmbeddings
- Runs a conversational loop, answering user questions using GPT-3.5-turbo
- Configuration and documentation improvements for maintainability

## Getting Started

1. Clone the repository
2. Install Python 3.11+ and required packages (see below)
3. Set your OpenAI API key as an environment variable (`OPENAI_API_KEY`) or in `constants.py`
4. Run the bot:
   ```powershell
   python main.py
   ```

## Requirements

- Python 3.11+
- `openai`, `langchain`, `chromadb`
- `unstructured` (for directory loading and `.xlsx`/other file types)

Install dependencies:

```powershell
pip install openai langchain chromadb
pip install "unstructured[xlsx]"
```

## File Structure

- `main.py` - Main application logic
- `constants.py` - Stores your OpenAI API key (if not using environment variable)
- `data/` - Contains metadata

## Example Usage

```
How can I assist you?: What's the price of a dozen of watermelons?
18.00
```

## License

Specify your license here (e.g., MIT, Apache-2.0).

---

Feel free to improve and extend AI-Assistant bot for your needs. Contributions are welcome!
