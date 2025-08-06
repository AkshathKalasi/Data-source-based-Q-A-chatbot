import os
import os
import uuid
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .chat_utils import calculate_answer_confidence


from typing import Optional, Dict, Any


# Initialize checkpointer
from langgraph.checkpoint.memory import MemorySaver

# Make checkpointer global so it persists across requests
_global_checkpointer = None

def get_checkpointer():
    global _global_checkpointer
    if _global_checkpointer is None:
        _global_checkpointer = MemorySaver()
    return _global_checkpointer




class ChatState(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    sources: List[str]
    confidence_score: float
    needs_clarification: bool
    ambiguity_info: dict
    context_choice: str
    thread_id: str
    chat_history: List[dict]


def detect_ambiguity(query: str, documents: List[Document]) -> Optional[Dict[str, Any]]:
    """Improved ambiguity detection based on document content analysis"""
    if not query:  # Add this check
        return None
        
    ambiguous_terms = {
        'apple': ['Apple Inc. technology/company', 'Apple fruit/food'],
        'java': ['Java programming language', 'Java island/coffee'],
        'python': ['Python programming language', 'Python snake/animal'],
        'amazon': ['Amazon company/AWS', 'Amazon rainforest/river'],
        'windows': ['Microsoft Windows OS', 'Windows architectural/glass']
    }
    
    query_lower = query.lower()
    for term, options in ambiguous_terms.items():
        if term in query_lower:
            if not documents or len(documents) == 0:
                return {'term': term, 'options': options, 'original_question': query}
            
            # Analyze document content for context clues
            doc_text = ' '.join([doc.page_content.lower() if doc.page_content else '' for doc in documents[:3]])
            
            # Better context scoring
            tech_keywords = ['programming', 'code', 'software', 'computer', 'technology', 'microsoft', 'company', 'business']
            nature_keywords = ['fruit', 'food', 'animal', 'nature', 'forest', 'tree', 'snake', 'island']
            
            tech_score = sum(1 for keyword in tech_keywords if keyword in doc_text)
            nature_score = sum(1 for keyword in nature_keywords if keyword in doc_text)
            
            # If scores are equal or both very low, trigger HITL
            if abs(tech_score - nature_score) <= 1 or (tech_score + nature_score) < 2:
                return {'term': term, 'options': options, 'original_question': query}
    
    return None


def retrieve_node(state: ChatState) -> ChatState:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=os.getenv("CHROMA_DB_PATH", "chroma_db"), embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        documents = retriever.invoke(state["query"])
        return {"documents": documents}
    except Exception:
        return {"documents": []}
    

def hitl_decision_node(state: ChatState) -> ChatState:
    # If user already provided context choice, skip clarification
    if state.get("context_choice"):
        return {"needs_clarification": False, "ambiguity_info": None}
    
    # Check for ambiguity
    ambiguity_info = detect_ambiguity(state["query"], state.get("documents", []))
    if ambiguity_info:
        return {"needs_clarification": True, "ambiguity_info": ambiguity_info}
    
    # Check confidence for low-quality responses
    if state.get("documents"):
        try:
            # Quick confidence check
            doc_relevance = sum(1 for doc in state["documents"][:3] 
                              if any(word in doc.page_content.lower() 
                                   for word in state["query"].lower().split()))
            if doc_relevance == 0:
                return {
                    "needs_clarification": True,
                    "ambiguity_info": {
                        'term': 'context',
                        'options': ['More specific information', 'General overview'],
                        'original_question': state["query"]
                    }
                }
        except:
            pass
    
    return {"needs_clarification": False, "ambiguity_info": None}



def generate_node(state: ChatState) -> ChatState:
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=300
        )
        
        context_text = "\n".join([doc.page_content[:400] for doc in state["documents"][:3]])
        
        # Build chat history context
        history_context = ""
        if state.get("chat_history"):
            recent_history = state["chat_history"][-4:]  # Last 4 messages
            for msg in recent_history:
                history_context += f"{msg['role']}: {msg['content'][:100]}...\n"
        
        if state.get("context_choice"):
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Context: {context_text}\n\nChat History:\n{history_context}\n\nUser wants: {state['context_choice']}. Focus on this aspect."),
                ("human", "{input}"),
            ])
        else:
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Context: {context_text}\n\nChat History:\n{history_context}\n\nAnswer based on context and conversation flow."),
                ("human", "{input}"),
            ])
        
        chain = qa_prompt | llm
        result = chain.invoke({"input": state["query"]})
        
        confidence_score = calculate_answer_confidence(result.content, state["documents"], state["query"])
        sources = [doc.metadata.get('source', 'N/A') for doc in state["documents"][:3]]
        
        return {
            "answer": result.content,
            "sources": sources,
            "confidence_score": float(confidence_score)
        }
    except Exception as e:
        return {"answer": f"Error processing request: {str(e)}", "sources": [], "confidence_score": 0.0}



def should_clarify(state: ChatState) -> str:
    return "clarify" if state.get("needs_clarification", False) else "generate"

def create_hitl_timetravel_graph():
    workflow = StateGraph(ChatState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("hitl_decision", hitl_decision_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "hitl_decision")
    workflow.add_conditional_edges("hitl_decision", should_clarify, {"clarify": END, "generate": "generate"})
    workflow.add_edge("generate", END)
    
    return workflow.compile(checkpointer=get_checkpointer())



def send_message_with_hitl_timetravel(message, thread_id=None, context_choice=None, chat_history=None):
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    try:
        graph = create_hitl_timetravel_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "query": message,
            "documents": [],
            "answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "needs_clarification": False,
            "ambiguity_info": None,
            "context_choice": context_choice,
            "chat_history": chat_history or []  # ADD THIS LINE
        }
        
        result = graph.invoke(initial_state, config)
        
        if result.get("needs_clarification", False):
            return {
                'needs_clarification': True,
                'ambiguity_clarification': result.get("ambiguity_info"),
                'thread_id': thread_id,
                'answer': "I need clarification to provide an accurate answer.",
                'sources': [],
                'confidence_score': 0.0
            }
        
        return {
            'answer': result["answer"],
            'sources': result["sources"],
            'confidence_score': result["confidence_score"],
            'thread_id': thread_id,
            'needs_clarification': False
        }
    except Exception as e:
        return {'answer': f"Error: {str(e)}", 'sources': [], 'confidence_score': 0.0, 'thread_id': thread_id, 'needs_clarification': False}



def send_message_with_timetravel(message, thread_id=None, context_choice=None):
    return send_message_with_hitl_timetravel(message, thread_id, context_choice)


def get_thread_history(thread_id):
    try:
        graph = create_hitl_timetravel_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        history = []
        for state in graph.get_state_history(config):
            history.append({
                'checkpoint_id': state.config['configurable'].get('checkpoint_id', 'unknown'),
                'step': state.metadata.get('step', 0),
                'node': state.metadata.get('source', 'unknown'),
                'timestamp': str(state.created_at) if state.created_at else None,
                'query': state.values.get('query', ''),
                'answer': state.values.get('answer', '')
            })
        return {'history': history}
    except Exception as e:
        return {'history': [], 'error': str(e)}


def rewind_to_checkpoint(thread_id, checkpoint_id):
    try:
        graph = create_hitl_timetravel_graph()
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        state = graph.get_state(config)
        return {'success': True, 'state': state.values}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def resume_from_checkpoint(thread_id, checkpoint_id, new_query):
    try:
        # Simple approach: just send a new message to continue the conversation
        result = send_message_with_hitl_timetravel(new_query, thread_id)
        return result
    except:
        # Completely safe fallback
        return {
            'answer': 'Resume function is not available at the moment', 
            'sources': [], 
            'confidence_score': 0.0,
            'needs_clarification': False
        }


