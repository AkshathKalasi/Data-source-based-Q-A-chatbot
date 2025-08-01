from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_view'),
    path('send_message/', views.send_message, name='send_message'),
    path('add_data_source/', views.add_data_source, name='add_data_source'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('get_chat_history/', views.get_chat_history, name='get_chat_history'),
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),
    path('request_human_help/', views.request_human_help, name='request_human_help'),

]
