from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from .chat_utils import *
import operator
import time

# ========== Enhanced caching for speed ==========
_vectorstore_cache = None
_embeddings_cache = None
_llm_8b_cache = None
_llm_70b_cache = None


def get_cached_vectorstore():
    global _vectorstore_cache, _embeddings_cache
    if _vectorstore_cache is None:
        print("🔄 Loading vectorstore (first time only)...")
        if _embeddings_cache is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        from langchain_chroma import Chroma
        _vectorstore_cache = Chroma(persist_directory=os.getenv("CHROMA_DB_PATH", "chroma_db"), embedding_function=_embeddings_cache)
    return _vectorstore_cache

def get_cached_llm(is_complex=False):
    global _llm_8b_cache, _llm_70b_cache
    from langchain_groq import ChatGroq
    
    if is_complex:
        if _llm_70b_cache is None:
            _llm_70b_cache = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-70b-8192",
                temperature=0.2,
                max_tokens=400,
                timeout=12,
                max_retries=1
            )
        return _llm_70b_cache
    else:
        if _llm_8b_cache is None:
            _llm_8b_cache = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-8b-8192",
                temperature=0.1,
                max_tokens=250,
                timeout=8,
                max_retries=1
            )
        return _llm_8b_cache

class ChatState(TypedDict):
    query: str
    chat_history: List
    documents: Annotated[List[Document], operator.add]
    context: str
    metadata: Annotated[dict, operator.or_]
    answer: str
    sources: List[str]
    confidence_score: float
    query_sentiment: dict
    answer_sentiment: dict

def retrieve_documents_node(state: ChatState) -> ChatState:
    start_time = time.time()
    print("🔍 Node: Retrieving documents...")
    vectorstore = get_cached_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced to 3 for speed
    documents = retriever.invoke(state["query"])
    execution_time = time.time() - start_time
    print(f"   ⏱️  Retrieve: {execution_time:.3f}s")
    
    return {
        "documents": documents,
        "metadata": {"retrieval_complete": True, "retrieve_time": execution_time}
    }

def build_context_node(state: ChatState) -> ChatState:
    start_time = time.time()
    print("📝 Node: Building context...")
    history_context = ""
    if state["chat_history"]:
        recent_messages = list(state["chat_history"])[-2:]  # Reduced to 2
        for msg in recent_messages:
            history_context += f"{msg.role}: {msg.content}\n"
    execution_time = time.time() - start_time
    print(f"   ⏱️  Context: {execution_time:.3f}s")
    
    return {
        "context": history_context,
        "metadata": {"context_complete": True, "context_time": execution_time}
    }

def calculate_confidence_node(state: ChatState) -> ChatState:
    start_time = time.time()
    print("📊 Node: Pre-calculating confidence factors...")
    query_complexity = len(state["query"].split())
    execution_time = time.time() - start_time
    print(f"   ⏱️  Confidence: {execution_time:.3f}s")
    
    return {
        "metadata": {
            "confidence_factors": {"query_complexity": query_complexity},
            "confidence_complete": True,
            "confidence_time": execution_time
        }
    }

def analyze_sentiment_node(state: ChatState) -> ChatState:
    start_time = time.time()
    print("😊 Node: Analyzing query sentiment...")
    
    try:
        from textblob import TextBlob
        blob = TextBlob(state["query"][:150])  # Further reduced
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        sentiment = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        query_sentiment = {"sentiment": sentiment, "polarity": round(polarity, 3), "subjectivity": round(subjectivity, 3)}
    except:
        query_sentiment = {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}
    
    execution_time = time.time() - start_time
    print(f"   ⏱️  Sentiment: {execution_time:.3f}s")
    
    return {
        "query_sentiment": query_sentiment,
        "metadata": {"sentiment_complete": True, "sentiment_time": execution_time}
    }

def generate_answer_node(state: ChatState) -> ChatState:
    start_time = time.time()
    print("🧠 Node: Generating answer...")
    
    query_lower = state["query"].lower()
    is_complex = any(keyword in query_lower for keyword in 
                    ["explain in detail", "analyze deeply", "comprehensive", "elaborate", "compare in detail"])
    
    print(f"   🤖 Using {'70B' if is_complex else '8B'} model")
    
    llm = get_cached_llm(is_complex)
    
    top_docs = state["documents"][:2]  # Use only top 2
    context_text = "\n".join([doc.page_content[:500] for doc in top_docs])  # Reduced to 500
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Context: {context_text}\n\nPrevious: {state['context'][:150]}\n\nAnswer concisely."),
        ("human", "{input}"),
    ])
    
    chain = qa_prompt | llm
    result = chain.invoke({"input": state["query"]})
    
    confidence_score = calculate_answer_confidence(result.content, top_docs, state["query"])
    sources = [doc.metadata.get('source', 'N/A') for doc in top_docs]
    
    try:
        from textblob import TextBlob
        blob = TextBlob(result.content[:200])
        polarity = blob.sentiment.polarity
        sentiment = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        answer_sentiment = {"sentiment": sentiment, "polarity": round(polarity, 3), "subjectivity": round(blob.sentiment.subjectivity, 3)}
    except:
        answer_sentiment = {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}
    
    execution_time = time.time() - start_time
    print(f"   ⏱️  Generate: {execution_time:.3f}s")
    
    return {
        "answer": result.content,
        "sources": sources,
        "confidence_score": confidence_score,
        "answer_sentiment": answer_sentiment,
        "metadata": {"generate_time": execution_time}
    }

def create_chat_graph():
    workflow = StateGraph(ChatState)
    
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("context", build_context_node)
    workflow.add_node("confidence", calculate_confidence_node)
    workflow.add_node("sentiment", analyze_sentiment_node)
    workflow.add_node("generate", generate_answer_node)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_edge("retrieve", "context")
    workflow.add_edge("retrieve", "confidence") 
    workflow.add_edge("retrieve", "sentiment")
    workflow.add_edge("context", "generate")
    workflow.add_edge("confidence", END)
    workflow.add_edge("sentiment", END)
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def get_chat_response_langgraph(prompt, chat_history):
    try:
        overall_start = time.time()  # Fixed variable name
        graph = create_chat_graph()
        
        initial_state = {
            "query": prompt,
            "chat_history": chat_history,
            "documents": [],
            "context": "",
            "metadata": {},
            "confidence_score": 0.0,
            "answer": "",
            "sources": [],
            "query_sentiment": {},
            "answer_sentiment": {}
        }
        
        print(f"🚀 Starting hybrid execution for: {prompt[:50]}...")
        result = graph.invoke(initial_state)
        
        total_time = time.time() - overall_start  # Fixed variable name
        print(f"✅ Hybrid execution completed in {total_time:.3f}s!")
        
        return {
            'answer': result["answer"],
            'sources': result["sources"],
            'is_uncertain': is_uncertain_answer(result["answer"], result["confidence_score"]),
            'confidence_score': result["confidence_score"],
            'retrieved_docs': len(result["documents"]),
            'relevant_docs': len(result["documents"]),
            'query_sentiment': result["query_sentiment"],
            'answer_sentiment': result["answer_sentiment"]
        }
        
    except Exception as e:
        print(f"❌ LangGraph ERROR: {str(e)}")
        return {
            'answer': "Error occurred. Please try again.",
            'sources': [],
            'is_uncertain': True,
            'confidence_score': 0.0,
            'retrieved_docs': 0,
            'relevant_docs': 0,
            'query_sentiment': {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0},
            'answer_sentiment': {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}
        }
