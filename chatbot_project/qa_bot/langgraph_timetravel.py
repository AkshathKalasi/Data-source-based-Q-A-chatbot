import os
import uuid
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .chat_utils import calculate_answer_confidence


from .time_travel_engine import TimeTravelEngine
from .session_tree_manager import SessionTreeManager
from .models import SessionTree, NodeExecution, ExecutionCheckpoint



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
    if not query:
        return None
        
    ambiguous_terms = {
        'apple': ['Apple Inc. technology/company', 'Apple fruit/food'],
        'java': ['Java programming language', 'Java island/coffee'],
        'python': ['Python programming language', 'Python snake/animal'],
        'amazon': ['Amazon company/AWS', 'Amazon rainforest/river'],
        'windows': ['Microsoft Windows OS', 'Windows architectural/glass'],
        'lotus': ['Lotus car company', 'Lotus flower/plant'],
        'apollo': ['Apollo space mission/NASA', 'Apollo tires company'],
        'mercury': ['Mercury planet', 'Mercury chemical element'],
        'mars': ['Mars planet', 'Mars chocolate company'],
        'oracle': ['Oracle database/company', 'Oracle ancient prophecy'],
        'shell': ['Shell oil company', 'Shell sea creature'],
        'ford': ['Ford car company', 'Ford river crossing'],
        'tesla': ['Tesla car company', 'Tesla scientist Nikola'],
        'jaguar': ['Jaguar car company', 'Jaguar animal'],
        'puma': ['Puma sportswear brand', 'Puma animal'],
        'nike': ['Nike sportswear brand', 'Nike Greek goddess'],
        'corona': ['Corona beer', 'Corona virus/pandemic'],
        'delta': ['Delta airlines', 'Delta river formation'],
        'target': ['Target retail store', 'Target aim/goal'],
        'mint': ['Mint plant/herb', 'Mint money/currency']
    } 
    
    query_lower = query.lower()
    
    # Check if query already has clear context indicators
    context_indicators = ['programming', 'language', 'code', 'software', 'technology', 'company', 'fruit', 'animal', 'food']
    has_context = any(indicator in query_lower for indicator in context_indicators)
    
    for term, options in ambiguous_terms.items():
        if term in query_lower and not has_context:
            # Only trigger HITL if no clear context in query and no documents
            if not documents or len(documents) == 0:
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
    # If user already provided context choice, skip clarification completely
    if state.get("context_choice"): 
        return {"needs_clarification": False, "ambiguity_info": None}
    
    # Only check for ambiguity if no context is provided
    ambiguity_info = detect_ambiguity(state["query"], state.get("documents", []))
    if ambiguity_info:
        return {"needs_clarification": True, "ambiguity_info": ambiguity_info}
    
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



def execute_with_time_travel(message, thread_id=None, rerun_nodes=None, context_choice=None, chat_history=None):
    """Execute pipeline with time travel support"""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # Initialize engines
    tt_engine = TimeTravelEngine()
    tree_manager = SessionTreeManager()
    
    # Create question node in session tree
    question_node_id = tree_manager.create_session_node(
        thread_id=thread_id,
        content=message,
        node_type='question',
        metadata={'context_choice': context_choice}
    )
    
    # Define pipeline nodes
    pipeline_nodes = {
        'retrieve': retrieve_node,
        'hitl_decision': hitl_decision_node,
        'generate': generate_node
    }
    
    initial_state = {
        "query": message,
        "documents": [],
        "answer": "",
        "sources": [],
        "confidence_score": 0.0,
        "needs_clarification": False,
        "ambiguity_info": None,
        "context_choice": context_choice,
        "chat_history": chat_history or []
    }
    
    # Execute with time travel
    result = tt_engine.time_travel_execute(
        pipeline_nodes=pipeline_nodes,
        initial_state=initial_state,
        thread_id=thread_id,
        rerun_nodes=rerun_nodes or set()
    )
    
    # Create response node in session tree
    response_node_id = tree_manager.create_session_node(
        thread_id=thread_id,
        content=result.get('answer', ''),
        node_type='response',
        parent_id=question_node_id,
        metadata={
            'sources': result.get('sources', []),
            'confidence_score': result.get('confidence_score', 0.0),
            'execution_log': result.get('_execution_log', [])
        }
    )
    
    return {
        'answer': result.get('answer', ''),
        'sources': result.get('sources', []),
        'confidence_score': result.get('confidence_score', 0.0),
        'thread_id': thread_id,
        'question_node_id': question_node_id,
        'response_node_id': response_node_id,
        'execution_log': result.get('_execution_log', []),
        'needs_clarification': result.get('needs_clarification', False)
    }

def rerun_specific_nodes(thread_id: str, node_names: List[str], new_query: str = None):
    """Re-run specific nodes in pipeline"""
    try:
        # Get last execution state
        tree_manager = SessionTreeManager()
        tree = tree_manager.get_session_tree(thread_id)
        
        if not tree['tree']:
            return {'error': 'No execution history found'}
        
        # Get last question for context
        last_question = None
        for node in reversed(tree['tree']):
            if node['type'] == 'question':
                last_question = node['content']
                break
        
        query = new_query or last_question or "Previous query"
        
        return execute_with_time_travel(
            message=query,
            thread_id=thread_id,
            rerun_nodes=set(node_names)
        )
    except Exception as e:
        return {'error': str(e)}

def get_session_tree_view(thread_id: str):
    """Get session tree for visualization"""
    tree_manager = SessionTreeManager()
    return tree_manager.get_session_tree(thread_id)

def create_branch_from_node(node_id: str, new_query: str):
    """Create new branch from existing node"""
    tree_manager = SessionTreeManager()
    branch_node_id = tree_manager.create_branch(node_id, new_query)
    
    if branch_node_id:
        # Execute new branch
        node = SessionTree.objects.get(session_id=node_id)
        new_thread_id = f"{node.thread_id}_branch_{uuid.uuid4().hex[:8]}"
        
        return execute_with_time_travel(
            message=new_query,
            thread_id=new_thread_id
        )
    
    return {'error': 'Failed to create branch'}



def get_execution_metrics(thread_id):
    """Get execution metrics"""
    from .models import NodeExecution
    try:
        executions = NodeExecution.objects.filter(thread_id=thread_id)
        return [{'node': e.node_name, 'runtime_ms': e.runtime_ms} for e in executions]
    except:
        return []
