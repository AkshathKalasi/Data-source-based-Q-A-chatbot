from dotenv import load_dotenv
# Refactored Version of the QA Chatbot Pipeline with Callback Handler and pdfplumber for PDF Parsing
# Added Chat History and Context Awareness

import os
import logging
from urllib.parse import urlparse, unquote
from typing import Any, List, Dict
import wikipedia
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # use updated module
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Added MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory # Added ChatMessageHistory

from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Constants
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")

# ========== STYLING ==========
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ========== CALLBACK HANDLER ==========
class MyCustomCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to print detailed execution steps."""
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any:
        print(f"\n{bcolors.OKBLUE}[Retriever Start] Searching for documents with query: '{query}'{bcolors.ENDC}")

    def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> Any:
        print(f"{bcolors.OKCYAN}[Retriever End] Found {len(documents)} documents.{bcolors.ENDC}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print(f"\n{bcolors.OKGREEN}[LLM Start] Sending prompt to model...{bcolors.ENDC}")
        # Print only the first 500 characters of the first prompt for brevity
        print(f"{bcolors.WARNING}{prompts[0][:500]}...{bcolors.ENDC}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        print(f"{bcolors.OKGREEN}[LLM End] Received response from model.{bcolors.ENDC}")

# ========== EXTRACTORS ==========
def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path.strip()) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text, file_path
    except Exception as e:
        logging.warning(f"[PDF Error] {file_path} ? {e}")
        return "", file_path

def extract_text_from_txt(file_path):
    try:
        with open(file_path.strip(), 'r', encoding='utf-8') as f:
            return f.read(), file_path
    except Exception as e:
        logging.warning(f"[TXT Error] {file_path} ? {e}")
        return "", file_path

def extract_text_from_wikipedia_url(url):
    try:
        title = unquote(urlparse(url).path.split("/wiki/")[-1])
        page = wikipedia.page(title, auto_suggest=False)
        return page.content, url
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning(f"[Wikipedia Disambiguation] {url} ? {e.options[:3]}...")
        return "", url
    except Exception as e:
        logging.warning(f"[Wikipedia Error] {url} ? {e}")
        return "", url

# ========== DATA INGESTION UTILS ==========
def import_data_batch(source_type, source_input):
    sources = source_input.split(",")
    all_documents = []

    for src in sources:
        if source_type == "wikipedia":
            text, meta = extract_text_from_wikipedia_url(src)
        elif source_type == "txt":
            text, meta = extract_text_from_txt(src)
        elif source_type == "pdf":
            text, meta = extract_text_from_pdf(src)
        else:
            raise ValueError("Unsupported source_type. Use 'wikipedia', 'txt', or 'pdf'.")

        if text.strip():
            all_documents.append(Document(page_content=text, metadata={"source": meta}))

    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

def store_chunks_in_chroma(chunks, persist_dir=CHROMA_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    logging.info(f"? Stored {len(chunks)} chunks in ChromaDB at '{persist_dir}'")
    return vectorstore

def load_vectorstore(persist_dir=CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# ========== DATA INGESTION ENTRY POINT ==========
def run_data_ingestion():
    print("\n?? Add new data:")
    source_type = input("Enter source type (wikipedia/txt/pdf): ").strip().lower()
    if source_type not in ["wikipedia", "txt", "pdf"]:
        print("? Invalid source type.")
        return

    source_input = input("Enter source(s) (comma-separated paths or URLs): ").strip()
    documents = import_data_batch(source_type, source_input)

    if not documents:
        print("? No documents found. Skipping.")
        return

    print("?? Splitting text into chunks...")
    chunks = split_documents(documents)

    print("?? Storing chunks into vector DB...")
    store_chunks_in_chroma(chunks)

# ========== QA CHATBOT ==========

def get_query_complexity(query):
    """Determine if query needs detailed response"""
    complex_keywords = [
        "explain", "detail", "how", "why", "describe", "elaborate", 
        "comprehensive", "analysis", "compare", "difference", "tell me about"
    ]
    return any(keyword in query.lower() for keyword in complex_keywords)

def launch_qa_bot():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    callback_handler = MyCustomCallbackHandler()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Use the provided context to give accurate answers.

Context: {context}

Instructions:
- Answer based on the context provided
- Be comprehensive for detailed questions
- Be concise for simple questions
- Say "I don't have enough information" if context is insufficient"""),
        ("human", "{input}"),
    ])

    chat_history = ChatMessageHistory()
    print("\n🤖 Chatbot ready! Using Groq API with smart model selection.")
    
    while True:
        query = input("\n🗣️ You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            # Smart model selection
            is_complex = get_query_complexity(query)
            
            if is_complex:
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama3-70b-8192",  # 70B for complex queries
                    temperature=0.3,
                    max_tokens=500
                )
                print("🧠 Using Llama-70B for detailed response...")
            else:
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama3-8b-8192",   # 8B for simple queries
                    temperature=0.3,
                    max_tokens=200
                )
                print("⚡ Using Llama-8B for quick response...")

            document_chain = create_stuff_documents_chain(llm, qa_prompt)
            qa_chain = create_retrieval_chain(retriever, document_chain)

            result = qa_chain.invoke({"input": query})
            answer = result['answer']
            print(f"\n💡 Answer: {answer}")

            chat_history.add_user_message(query)
            chat_history.add_ai_message(answer)

            print("\n📚 Sources:")
            for i, doc in enumerate(result.get("context", [])):
                print(f" {i+1}. {doc.metadata.get('source', 'N/A')}")

            satisfied = input("\n✅ Satisfactory? (y/n): ").strip().lower()
            if satisfied == "n":
                print("📝 Add more data to improve answers.")
                run_data_ingestion()

        except Exception as e:
            logging.error(f"❌ Error: {e}")
            print("An error occurred. Please try again.")


# ========== MAIN PIPELINE ==========
def run_pipeline():
    print("?? Launching chatbot...")
    try:
        launch_qa_bot()
    except Exception as e:
        logging.critical(f"?? Critical error in chatbot: {e}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    run_pipeline()
