import streamlit as st
import uuid
import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_community.document_loaders import WebBaseLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

MONGODB_STR = os.getenv("MONGODB_STR")
if not MONGODB_STR:
    raise ValueError("MONGODB_STR environment variable not set")

# Connect to MongoDB
mongo_client = MongoClient(MONGODB_STR)
db = mongo_client.get_database("gemmastream_rag")
collection = db.get_collection("embeddings")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LangSmith Tracing ---
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    logger.info(f"LangSmith tracing enabled for project: {project_name}")

st.set_page_config(
    page_title="site-rag",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_url" not in st.session_state:
    st.session_state.current_url = None

# --- Helper Functions ---

def get_vectorstore_from_url(url: str, chunk_size: int, chunk_overlap: int) -> MongoDBAtlasVectorSearch:
    """
    Loads a URL, splits content, and stores embeddings.
    """
    try:
        with st.status("📥 Loading website content...", expanded=False) as status:
            logger.info(f"Loading URL: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            if not docs:
                raise ValueError("No content found at the provided URL")
            
            status.update(label="✅ Website loaded successfully!", state="complete")
        
        with st.status(f"✂️ Splitting content (Size: {chunk_size}, Overlap: {chunk_overlap})...", expanded=False) as status:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            splits = text_splitter.split_documents(docs)
            
            logger.info(f"Created {len(splits)} chunks from {len(docs)} documents")
            status.update(label=f"✅ Split into {len(splits)} chunks!", state="complete")
        
        with st.status("🧠 Creating embeddings and storing in MongoDB...", expanded=False) as status:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
            
            # Create vector store
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=splits,
                embedding=embeddings,
                collection=collection,
                index_name="vector_index"
            )
            logger.info("Vector store created and stored in MongoDB successfully")
            status.update(label="✅ Vector store created in MongoDB!", state="complete")
        
        return vector_store
        
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        st.error(f"❌ Content Error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error loading URL: {str(e)}")
        st.error(f"❌ Failed to load website: {str(e)}")
        raise


def get_retrieval_chain(vector_store: MongoDBAtlasVectorSearch):
    """
    Creates a simple retrieval chain with strict context adherence.
    """
    try:
        # 1. Select LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OpenAI API Key not found in environment variables.")
            raise ValueError("OpenAI API Key missing")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

        # 2. Retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 30
            }
        )

        # 3. Define Simple Answer Prompt with STRICT Instruction
        prompt_answer = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. 
            Answer the user's question based ONLY on the provided context below.
            If the answer is NOT in the context, simply state: "I cannot find the answer in the provided website content."
            DO NOT using any outside knowledge.
            
            Context:
            {context}
            """),
            ("user", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(llm, prompt_answer)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        logger.error(f"Error creating retriever chain: {str(e)}")
        st.error(f"❌ Failed to initialize AI model: {str(e)}")
        raise


def process_user_query(query: str, chain):
    try:
        logger.info(f"Processing query: {query}")
        
        response = chain.invoke({
            "input": query
        })
        answer = response.get("answer", "Unable to generate response")
        
        logger.info("Query processed successfully")
        return answer, response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"❌ Error processing your query: {str(e)}")
        raise


def clear_local_data():
    """Clear local session state vectors."""
    try:
        st.session_state.vector_store = None
        st.session_state.current_url = None
        st.success("Local context cleared.")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing local context: {str(e)}")


def main():
    st.title("site-rag")
    st.markdown("Ask questions about any website.")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.header("Configuration")

        # --- Chunking Control ---
        st.subheader("Chunking Settings")
        chunk_size = st.number_input(
            label="Chunk Size",
            min_value=200,
            max_value=5000,
            value=1000,
            step=50
        )
        chunk_overlap = int(chunk_size * 0.2)
        st.caption(f"Overlap: {chunk_overlap} chars")

        st.divider()

        url_input = st.text_input(
            label="Website URL",
            placeholder="https://example.com",
        )
        
        if st.button("Process Website", use_container_width=True):
             if not url_input.strip():
                st.error("Please enter a valid URL")
             else:
                try:
                    with st.spinner("Processing website..."):
                        vector_store = get_vectorstore_from_url(url_input, chunk_size, chunk_overlap)
                        st.session_state.vector_store = vector_store
                        st.session_state.current_url = url_input
                    st.success(f"Indexed: {url_input}")
                except Exception as e:
                    logger.error(f"Failed: {str(e)}")

        if st.session_state.current_url:
            st.success(f"Active: {st.session_state.current_url}")
        else:
            st.info("No website indexed")
        
        st.divider()
        
        if st.button("clear database", use_container_width=True):
            clear_local_data()

    # --- Main Logic ---

    if st.session_state.vector_store is not None:
        
        # Simple Input Area
        user_query = st.chat_input(placeholder="Ask a question about the content...")
        
        if user_query:
            # Display user question
            st.markdown(f"**You:** {user_query}")
            
            try:
                # 1. Get Chain
                chain = get_retrieval_chain(st.session_state.vector_store)
                
                # 2. Generate Response
                with st.spinner("Generating answer..."):
                    answer, full_response = process_user_query(user_query, chain)
                
                # 3. Display Answer
                st.markdown(f"**AI:** {answer}")
                

                            
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                st.error("Failed to generate response.")
    else:
        st.info("👈 Enter a URL in the sidebar to begin.")


if __name__ == "__main__":
    main()
