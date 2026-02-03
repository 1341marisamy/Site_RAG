# Site_RAG

Streamlit app that builds a site-specific RAG (retrieval-augmented generation) pipeline. It loads a website, splits and embeds its content, stores vectors in MongoDB, and exposes a retrieval + chat interface to ask questions about the site.

## Features
- Load website content via WebBaseLoader
- Chunk text with RecursiveCharacterTextSplitter
- Create embeddings using OpenAI embeddings (text-embedding-3-large)
- Store and query vectors in MongoDB via MongoDBAtlasVectorSearch
- Streamlit UI with progress/status updates
- Optional LangSmith tracing (when LANGCHAIN_API_KEY is set)

## Files
- `app.py` — Streamlit application, ingestion and query logic
- `requirements.txt` — Python dependency list
- `.streamlit/` — Streamlit config (present)
- `.gitignore`

## Requirements
- Python 3.9+
- MongoDB Atlas (or accessible MongoDB) with write access
- OpenAI credentials (or configured model provider expected by langchain_openai)
- See `requirements.txt` for Python packages; install via pip

## Environment variables
- `MONGODB_STR` (required) — MongoDB connection string (used to connect to the `gemmastream_rag` DB and `embeddings` collection)
- `OPENAI_API_KEY` (recommended) — OpenAI API key used by OpenAIEmbeddings and ChatOpenAI
- `LANGCHAIN_API_KEY` (optional) — enables LangSmith tracing if present
- `LANGCHAIN_PROJECT` (optional) — project name for LangSmith tracing

Example `.env`:
```
MONGODB_STR="mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
OPENAI_API_KEY="sk-..."
# Optional
LANGCHAIN_API_KEY="langchain_..."
LANGCHAIN_PROJECT="site-rag"
```

## Installation
1. Clone the repo:
   git clone https://github.com/1341marisamy/Site_RAG.git
   cd Site_RAG

2. Install dependencies:
   pip install -r requirements.txt

3. Create a `.env` file with the required environment variables (see above).

## Running
Start the Streamlit app:
   streamlit run app.py

Default behavior:
- Enter a website URL in the UI.
- Configure chunk size and overlap (UI).
- The app will load the website, split the text into chunks, compute embeddings, store them in MongoDB, and create a vector store.
- Use the chat UI to ask questions; the app will retrieve supporting documents and use the Chat LLM to answer.

## Notes & Operational tips
- Database/collection: the app uses the database `gemmastream_rag` and collection `embeddings`. To reset ingestion, drop or clear the `embeddings` collection in your MongoDB.
- If WebBaseLoader returns "No content found", try a different URL or ensure the site allows scraping.
- Long-running operations show Streamlit status messages; check logs for more details.
- If you plan to re-ingest the same site multiple times, consider the behavior of the vector store and your index settings in MongoDB.
- If you use local models (e.g., ollama) or other providers, ensure environment and langchain configuration match those providers.

## Troubleshooting
- MONGODB_STR not set: app raises an error on startup — ensure your `.env` or environment contains `MONGODB_STR`.
- Authentication errors from OpenAI: verify `OPENAI_API_KEY` and network connectivity.
- Missing packages: run pip install -r requirements.txt; version conflicts may require creating a virtual environment.

## Security & Cost
- OpenAI embedding and chat usage may incur cost — monitor usage and rate limits.
- Do not commit secret keys to the repository. Use `.env` and keep it out of version control.

## Contributing
Contributions, issues, and feature requests are welcome. Please open an issue or submit a PR.

## License
Add a license (e.g., MIT) or update this section with your chosen license.