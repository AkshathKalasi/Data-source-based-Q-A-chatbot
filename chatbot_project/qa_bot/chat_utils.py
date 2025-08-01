from dotenv import load_dotenv
import os
import logging
from urllib.parse import urlparse, unquote
import wikipedia
import pdfplumber
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, List, Dict
import datetime

class TerminalColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TerminalCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        self.start_time = datetime.datetime.now()
        print(f"\n{TerminalColors.HEADER}🚀 CHATBOT EXECUTION STARTED{TerminalColors.ENDC}")
        print(f"{TerminalColors.CYAN}📝 Query: {inputs.get('input', 'N/A')[:100]}{TerminalColors.ENDC}")
        
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any:
        print(f"{TerminalColors.BLUE}🔍 Searching database for: '{query[:50]}...'{TerminalColors.ENDC}")
        
    def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> Any:
        print(f"{TerminalColors.GREEN}✅ Found {len(documents)} documents{TerminalColors.ENDC}")
            
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print(f"{TerminalColors.YELLOW}🧠 Generating response...{TerminalColors.ENDC}")
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        print(f"{TerminalColors.GREEN}✅ Response generated{TerminalColors.ENDC}")
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.start_time:
            duration = (datetime.datetime.now() - self.start_time).total_seconds()
            print(f"{TerminalColors.GREEN}🎉 Completed in {duration:.2f}s{TerminalColors.ENDC}\n")


load_dotenv()
logging.basicConfig(level=logging.INFO)

CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for relevant documents
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for certain answers

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text, file_path
    except Exception as e:
        return "", file_path

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), file_path
    except Exception as e:
        return "", file_path

def extract_text_from_wikipedia_url(url):
    try:
        title = unquote(urlparse(url).path.split("/wiki/")[-1])
        page = wikipedia.page(title, auto_suggest=False)
        return page.content, url
    except Exception as e:
        return "", url

def extract_text_from_web_url(url):
    try:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text from paragraphs, headers, and list items
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        text = "\n".join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])
        return text, url
    except Exception as e:
        return "", url

def import_data_batch(source_type, source_input):
    sources = source_input.split(",")
    all_documents = []

    for src in sources:
        src = src.strip()
        if not src:
            continue
            
        if source_type == "wikipedia":
            text, meta = extract_text_from_wikipedia_url(src)
        elif source_type == "txt":
            text, meta = extract_text_from_txt(src)
        elif source_type == "pdf":
            text, meta = extract_text_from_pdf(src)
        elif source_type == "web":
            text, meta = extract_text_from_web_url(src)
        else:
            raise ValueError("Unsupported source_type")

        if text.strip():
            all_documents.append(Document(page_content=text, metadata={"source": meta}))

    return all_documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced for better precision
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(documents)

def store_chunks_in_chroma(chunks):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def calculate_answer_confidence(answer, retrieved_docs, query):
    """Calculate confidence score based on multiple factors"""
    confidence_score = 0.0
    
    # Factor 1: Answer length and completeness
    if len(answer.split()) > 5:
        confidence_score += 0.2
    
    # Factor 2: Presence of specific information
    if any(keyword in answer.lower() for keyword in ['because', 'therefore', 'according to', 'based on']):
        confidence_score += 0.2
    
    # Factor 3: Document relevance using TF-IDF similarity
    if retrieved_docs:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            doc_texts = [doc.page_content for doc in retrieved_docs]
            doc_texts.append(query)
            
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            
            if max_similarity > SIMILARITY_THRESHOLD:
                confidence_score += 0.4
            else:
                confidence_score += max_similarity * 0.4
                
        except Exception:
            confidence_score += 0.1
    
    # Factor 4: Absence of uncertainty phrases
    uncertain_phrases = [
        "i don't know", "i'm not sure", "i cannot", "no information",
        "not mentioned", "unclear", "uncertain", "maybe", "possibly"
    ]
    
    if not any(phrase in answer.lower() for phrase in uncertain_phrases):
        confidence_score += 0.2
    
    return min(confidence_score, 1.0)

def is_uncertain_answer(answer, confidence_score=None):
    """Enhanced uncertainty detection"""
    # Pattern-based detection
    uncertain_patterns = [
        r"i don'?t know", r"i don'?t have", r"i'?m not sure",
        r"i do not know", r"i do not have", r"no information",
        r"cannot provide", r"don'?t have enough", r"insufficient",
        r"no context", r"no data", r"not mentioned", r"not provided",
        r"unclear", r"uncertain", r"i cannot", r"unable to"
    ]
    
    answer_lower = answer.lower()
    pattern_uncertain = any(re.search(pattern, answer_lower) for pattern in uncertain_patterns)
    
    # Confidence-based detection
    confidence_uncertain = confidence_score is not None and confidence_score < CONFIDENCE_THRESHOLD
    
    # Short answer detection (likely incomplete)
    short_answer = len(answer.split()) < 3
    
    return pattern_uncertain or confidence_uncertain or short_answer

def rerank_documents(query, documents, top_k=3):
    """Rerank documents based on relevance to query"""
    if not documents:
        return documents
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        doc_texts = [doc.page_content for doc in documents]
        doc_texts.append(query)
        
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        
        # Filter documents below similarity threshold
        relevant_indices = [i for i, sim in enumerate(similarities) if sim >= SIMILARITY_THRESHOLD]
        
        if not relevant_indices:
            # If no documents meet threshold, take top 1 with highest similarity
            relevant_indices = [np.argmax(similarities)]
        
        # Sort by similarity and take top_k
        relevant_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        return [documents[i] for i in relevant_indices]
        
    except Exception:
        return documents[:top_k]


   #  ===== Debugging utilities =======================================
    
def debug_vectorstore():
    vectorstore = load_vectorstore()
    # Check if vectorstore has any data
    try:
        test_results = vectorstore.similarity_search("AI artificial intelligence", k=5)
        print(f"Debug: Found {len(test_results)} documents for test query")
        for i, doc in enumerate(test_results):
            print(f"  {i+1}. Source: {doc.metadata.get('source', 'N/A')}")
            print(f"      Content preview: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Debug error: {e}")



# ========== sentiment analysis utilities ==========   No longer used for under 1 second response.

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Convert polarity to sentiment label
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3)
        }
    except ImportError:
        return {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}
    except Exception:
        return {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}


# ======================================================================

def get_chat_response(prompt, chat_history):
    try:
        from langchain_groq import ChatGroq
        
        callback_handler = TerminalCallbackHandler()
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        history_context = ""
        if chat_history:
            recent_messages = list(chat_history)[-4:]
            for msg in recent_messages:
                history_context += f"{msg.role}: {msg.content}\n"
        
        is_complex = any(keyword in prompt.lower() for keyword in 
                        ["explain", "detail", "how", "why", "describe"])
        
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192" if is_complex else "llama3-8b-8192",
            temperature=0.3,
            max_tokens=500 if is_complex else 200
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Previous conversation:
{history_context}

Use the context to answer accurately, considering the conversation history.

Context: {{context}}"""),
            ("human", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)
        
        result = qa_chain.invoke(
            {"input": prompt}, 
            config={"callbacks": [callback_handler]}
        )
        
        answer = result.get('answer', "I don't have an answer.")
        sources = [doc.metadata.get('source', 'N/A') for doc in result.get("context", [])]
        confidence_score = calculate_answer_confidence(answer, result.get("context", []), prompt)
        
        print(f"{TerminalColors.YELLOW}📊 Confidence: {confidence_score:.2f} | Sources: {len(sources)}{TerminalColors.ENDC}")
        
        return {
            'answer': answer,
            'sources': sources,
            'is_uncertain': is_uncertain_answer(answer, confidence_score),
            'confidence_score': confidence_score,
            'retrieved_docs': len(result.get("context", [])),
            'relevant_docs': len(result.get("context", []))
        }
        
    except Exception as e:
        print(f"{TerminalColors.RED}❌ ERROR: {str(e)}{TerminalColors.ENDC}")
        logging.error(f"Error: {e}")
        return {
            'answer': "Error occurred. Please try again.",
            'sources': [],
            'is_uncertain': True,
            'confidence_score': 0.0,
            'retrieved_docs': 0,
            'relevant_docs': 0
        }
