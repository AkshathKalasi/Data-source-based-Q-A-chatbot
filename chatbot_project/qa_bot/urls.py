from django.urls import path
from . import views

urlpatterns = [
    # Main chat interface
    path('', views.chat_view, name='chat_view'),
    
    # Time Travel endpoints
    path('send_message_timetravel/', views.send_message_timetravel, name='send_message_timetravel'),
    path('get_thread_history/', views.get_thread_history_view, name='get_thread_history'),
    path('rewind_thread/', views.rewind_thread, name='rewind_thread'),
    path('resume_thread/', views.resume_thread, name='resume_thread'),
    path('timetravel/', views.timetravel_view, name='timetravel_view'),
    
    # Regular chat endpoints
    path('send_message/', views.send_message, name='send_message'),
    path('add_data_source/', views.add_data_source, name='add_data_source'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('get_chat_history/', views.get_chat_history, name='get_chat_history'),
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),
    path('request_human_help/', views.request_human_help, name='request_human_help'),
    
    # Human review workflow
    path('generate_with_review/', views.generate_with_review, name='generate_with_review'),
    path('human_review_decision/', views.human_review_decision, name='human_review_decision'),
    path('refine_answer/', views.refine_answer, name='refine_answer'),
]
