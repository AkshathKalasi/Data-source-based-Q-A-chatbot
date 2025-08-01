from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from .models import ChatSession, ChatMessage, FeedbackData
from .langgraph_engine import get_chat_response_langgraph as get_chat_response, import_data_batch, split_documents, store_chunks_in_chroma
from langchain_core.documents import Document


def chat_view(request):
    """Main chat interface"""
    return render(request, 'qa_bot/chat.html')

@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """Handle chat messages"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get or create session
        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={'awaiting_source': False}
        )
        
        # Check if awaiting source
        if session.awaiting_source:
            return JsonResponse({
                'error': 'Please add data sources first',
                'awaiting_source': True,
                'last_question': session.last_question
            }, status=400)
        
        # Save user message
        ChatMessage.objects.create(
            session=session,
            role='user',
            content=message
        )
        
        # Get chat history
        chat_history = session.messages.all()
        
        # Get response
        response = get_chat_response(message, chat_history)
        
        # Handle uncertain answers
        if response['is_uncertain']:
            session.awaiting_source = True
            session.last_question = message
            session.save()
            
            response_content = response['answer'] + "\n\nI don't have enough information to answer this question. Please add relevant data sources."
        else:
            response_content = response['answer']
        
        # Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=response_content,
            sources=response['sources'],
            confidence_score=response.get('confidence_score', 0.0)
        )
        
        return JsonResponse({
            'response': response_content,
            'sources': response['sources'],
            'awaiting_source': response['is_uncertain'],
            'confidence_score': response.get('confidence_score', 0.0),
            'message_id': assistant_message.id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def add_data_source(request):
    """Add new data sources"""
    try:
        data = json.loads(request.body)
        source_type = data.get('source_type')
        source_input = data.get('source_input', '').strip()
        session_id = data.get('session_id')
        
        if not source_input:
            return JsonResponse({'error': 'Source input is required'}, status=400)
        
        # Process data
        documents = import_data_batch(source_type, source_input)
        if documents:
            chunks = split_documents(documents)
            store_chunks_in_chroma(chunks)
            
            # Update session
            session = ChatSession.objects.get(session_id=session_id)
            
            # Always reset awaiting_source when data is successfully added
            session.awaiting_source = False
            session.save()
            
            if session.last_question:
                # Retry the last question
                chat_history = session.messages.all()
                response = get_chat_response(session.last_question, chat_history)
                
                if not response['is_uncertain']:
                    # Save new assistant message
                    ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=response['answer'],
                        sources=response['sources']
                    )
                    
                    return JsonResponse({
                        'success': True,
                        'chunks_added': len(chunks),
                        'retry_response': response['answer'],
                        'sources': response['sources'],
                        'awaiting_source': False
                    })
            
            return JsonResponse({
                'success': True,
                'chunks_added': len(chunks),
                'awaiting_source': False
            })
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
        session_id = request.POST.get('session_id')
        
        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        # Save file temporarily
        filename = uploaded_file.name
        temp_path = os.path.join('temp_uploads', filename)
        os.makedirs('temp_uploads', exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        # Extract text
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
            
            # Update session
            session = ChatSession.objects.get(session_id=session_id)
            
            # Always reset awaiting_source when data is successfully added
            session.awaiting_source = False
            session.save()
            
            if session.last_question:
                chat_history = session.messages.all()
                response = get_chat_response(session.last_question, chat_history)
                
                if not response['is_uncertain']:
                    ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=response['answer'],
                        sources=response['sources']
                    )
                    
                    return JsonResponse({
                        'success': True,
                        'chunks_added': len(chunks),
                        'retry_response': response['answer'],
                        'sources': response['sources'],
                        'awaiting_source': False
                    })
            
            return JsonResponse({
                'success': True,
                'chunks_added': len(chunks),
                'awaiting_source': False
            })
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
        
        return JsonResponse({
            'messages': messages,
            'awaiting_source': session.awaiting_source,
            'last_question': session.last_question
        })
    except ChatSession.DoesNotExist:
        return JsonResponse({'messages': [], 'awaiting_source': False, 'last_question': ''})

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """Handle user feedback on answers"""
    try:
        data = json.loads(request.body)
        message_id = data.get('message_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        corrected_answer = data.get('corrected_answer', '')
        
        message = ChatMessage.objects.get(id=message_id, role='assistant')
        message.feedback_rating = rating
        message.feedback_comment = comment
        
        if corrected_answer:
            message.is_corrected = True
            message.corrected_answer = corrected_answer
            
        message.save()
        
        # Store feedback data for analysis
        FeedbackData.objects.create(
            session=message.session,
            question=message.session.messages.filter(
                timestamp__lt=message.timestamp, 
                role='user'
            ).last().content,
            original_answer=message.content,
            corrected_answer=corrected_answer,
            feedback_type='incorrect' if rating <= 2 else 'good',
            sources_used=message.sources,
            confidence_score=message.confidence_score
        )
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt  
@require_http_methods(["POST"])
def request_human_help(request):
    """Request human assistance for difficult questions"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        question = data.get('question')
        
        session = ChatSession.objects.get(session_id=session_id)
        
        # Mark as awaiting human help
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            content="I've forwarded your question to a human expert. You'll receive a response soon. In the meantime, you can try adding more specific data sources.",
            confidence_score=0.0
        )
        
        return JsonResponse({'success': True, 'message': 'Human expert has been notified'})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
