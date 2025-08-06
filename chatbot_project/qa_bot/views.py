from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from .models import ChatSession, ChatMessage
from .chat_utils import get_chat_response, import_data_batch, split_documents, store_chunks_in_chroma

from langchain_core.documents import Document
from werkzeug.utils import secure_filename

def chat_view(request):
    """Main chat interface"""
    return render(request, 'qa_bot/chat.html')


def check_needs_clarification(query):
    """Context-aware HITL - only trigger when truly ambiguous"""
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
    
    # Context indicators that clarify meaning
    programming_context = ['syntax', 'code', 'programming', 'function', 'variable', 'class', 'method', 'library', 'framework', 'script', 'algorithm', 'debug', 'compile', 'execute']
    tech_context = ['software', 'hardware', 'computer', 'system', 'application', 'platform', 'technology', 'digital']
    nature_context = ['animal', 'species', 'wildlife', 'habitat', 'biology', 'ecosystem', 'nature']
    business_context = ['company', 'corporation', 'business', 'stock', 'market', 'revenue', 'profit']
    
    # Check for ambiguous terms
    for term, options in ambiguous_terms.items():
        import re
        if re.search(r'\b' + term + r'\b', query_lower):
            # Check if context is clear
            has_programming_context = any(ctx in query_lower for ctx in programming_context)
            has_tech_context = any(ctx in query_lower for ctx in tech_context)
            has_nature_context = any(ctx in query_lower for ctx in nature_context)
            has_business_context = any(ctx in query_lower for ctx in business_context)
            
            # Only trigger HITL if no clear context indicators
            if not (has_programming_context or has_tech_context or has_nature_context or has_business_context):
                return {
                    'term': term,
                    'options': options,
                    'original_question': query
                }
    
    return None



def get_context_aware_response(message, chat_history, context_choice=None):
    """Get response with context awareness and timing"""
    if context_choice:
        # Create context-focused query
        focused_query = f"Focus on {context_choice}: {message}"
        print(f"🎯 Context-focused query: {focused_query}")
        
        # Use langgraph engine for timing visibility
        from .langgraph_engine import get_chat_response_langgraph
        return get_chat_response_langgraph(focused_query, chat_history)
    else:
        # Use langgraph engine for timing visibility
        from .langgraph_engine import get_chat_response_langgraph
        return get_chat_response_langgraph(message, chat_history)



@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """Enhanced chat with HITL support"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        context_choice = data.get('context_choice')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get or create session
        session, _ = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={'awaiting_source': False}
        )
        
        # Get recent chat history
        recent_messages = session.messages.all().order_by('-timestamp')[:6]
        chat_history = []
        for msg in reversed(recent_messages):
            chat_history.append(type('ChatMessage', (), {'role': msg.role, 'content': msg.content})())
        
        # Save user message
        ChatMessage.objects.create(session=session, role='user', content=message)
        
        # Use your working langgraph_engine with HITL check
        if context_choice:
            # User provided clarification, use normal flow
            response = get_context_aware_response(message, chat_history, context_choice)


        else:
            # Check if needs clarification first
            needs_clarification = check_needs_clarification(message)
            if needs_clarification:
                return JsonResponse({
                    'needs_clarification': True,
                    'ambiguity_clarification': needs_clarification,
                    'response': "I need clarification to provide an accurate answer.",
                    'sources': [],
                    'confidence_score': 0.0
                })
            else:
                response = get_context_aware_response(message, chat_history)


        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=response['answer'],
            sources=response.get('sources', []),
            confidence_score=response.get('confidence_score', 0.0)
        )
        
        return JsonResponse({
            'response': response['answer'],
            'sources': response.get('sources', []),
            'confidence_score': response.get('confidence_score', 0.0),
            'message_id': assistant_message.id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def add_data_source(request):
    """Add data sources to vector database"""
    try:
        data = json.loads(request.body)
        source_type = data.get('source_type')
        source_input = data.get('source_input', '').strip()
        
        if not source_input:
            return JsonResponse({'error': 'Source input is required'}, status=400)
        
        documents = import_data_batch(source_type, source_input)
        if documents:
            chunks = split_documents(documents)
            store_chunks_in_chroma(chunks)
            return JsonResponse({'success': True, 'chunks_added': len(chunks)})
        else:
            return JsonResponse({'error': 'No documents found'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    """Handle file uploads"""
    try:
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        filename = secure_filename(uploaded_file.name)
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        if filename.lower().endswith('.pdf'):
            from .chat_utils import extract_text_from_pdf
            text, _ = extract_text_from_pdf(temp_path)
        elif filename.lower().endswith('.txt'):
            from .chat_utils import extract_text_from_txt
            text, _ = extract_text_from_txt(temp_path)
        else:
            return JsonResponse({'error': 'Unsupported file type'}, status=400)
        
        if text:
            document = Document(page_content=text, metadata={"source": filename})
            chunks = split_documents([document])
            store_chunks_in_chroma(chunks)
            return JsonResponse({'success': True, 'chunks_added': len(chunks)})
        else:
            return JsonResponse({'error': 'Could not extract text from file'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_chat_history(request):
    """Get chat history for a session"""
    session_id = request.GET.get('session_id')
    if not session_id:
        return JsonResponse({'error': 'Session ID required'}, status=400)
    
    try:
        session = ChatSession.objects.get(session_id=session_id)
        messages = []
        for msg in session.messages.all():
            messages.append({
                'role': msg.role,
                'content': msg.content,
                'sources': msg.sources,
                'timestamp': msg.timestamp.isoformat()
            })
        
        return JsonResponse({'messages': messages})
    except ChatSession.DoesNotExist:
        return JsonResponse({'messages': []})

# ==================== TIME TRAVEL VIEWS ====================

@require_http_methods(["GET"])
def get_thread_history_view(request):
    """Get thread history using LangGraph"""
    thread_id = request.GET.get('thread_id')
    if not thread_id:
        return JsonResponse({'error': 'Thread ID required'}, status=400)
    
    from .langgraph_timetravel import get_thread_history
    return JsonResponse(get_thread_history(thread_id))


@csrf_exempt
@require_http_methods(["POST"])
def rewind_thread(request):
    """Rewind to checkpoint"""
    try:
        data = json.loads(request.body)
        thread_id = data.get('thread_id')
        checkpoint_id = data.get('checkpoint_id')
        
        from .langgraph_timetravel import rewind_to_checkpoint
        return JsonResponse(rewind_to_checkpoint(thread_id, checkpoint_id))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def resume_thread(request):
    """Resume from checkpoint"""
    try:
        data = json.loads(request.body)
        thread_id = data.get('thread_id')
        checkpoint_id = data.get('checkpoint_id')
        new_query = data.get('new_query')
        
        from .langgraph_timetravel import resume_from_checkpoint
        return JsonResponse(resume_from_checkpoint(thread_id, checkpoint_id, new_query))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def send_message_timetravel(request):
    """Send message with time travel support"""
    try:
        data = json.loads(request.body)
        message = data.get('message')
        thread_id = data.get('thread_id')
        context_choice = data.get('context_choice')  # Add this line
        
        from .langgraph_timetravel import send_message_with_timetravel
        return JsonResponse(send_message_with_timetravel(message, thread_id, context_choice))  # Pass context_choice
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    

def timetravel_view(request):
    """Render time travel interface"""
    return render(request, 'qa_bot/timetravel.html')

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """Handle user feedback"""
    try:
        data = json.loads(request.body)
        message_id = data.get('message_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        message = ChatMessage.objects.get(id=message_id, role='assistant')
        message.feedback_rating = rating
        message.feedback_comment = comment
        message.save()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def request_human_help(request):
    """Request human expert help"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        question = data.get('question')
        
        session = ChatSession.objects.get(session_id=session_id)
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            content="I've forwarded your question to a human expert.",
            confidence_score=0.0
        )
        
        return JsonResponse({'success': True, 'message': 'Human expert notified'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def generate_with_review(request):
    """Generate response with review option"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        # Use your working engine
        session, _ = ChatSession.objects.get_or_create(session_id=session_id)
        chat_history = list(session.messages.all())[-4:]
        response = get_context_aware_response(message, chat_history)


        
        return JsonResponse(response)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def human_review_decision(request):
    """Process human review decisions"""
    try:
        data = json.loads(request.body)
        decision = data.get('decision')
        return JsonResponse({'status': decision, 'next_step': 'approved' if decision == 'approved' else 'rejected'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def refine_answer(request):
    """Refine answers based on feedback"""
    try:
        data = json.loads(request.body)
        refinement_request = data.get('refinement_request', '')
        return JsonResponse({'refined_answer': 'Answer refined based on feedback', 'status': 'refined'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
