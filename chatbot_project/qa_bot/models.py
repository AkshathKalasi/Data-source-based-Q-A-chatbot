from django.db import models
import uuid

class ChatSession(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    awaiting_source = models.BooleanField(default=False)
    last_question = models.TextField(blank=True)

class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # ADD THESE NEW FIELDS
    confidence_score = models.FloatField(default=0.0)
    feedback_rating = models.IntegerField(null=True, blank=True, choices=[(1, 'Poor'), (2, 'Fair'), (3, 'Good'), (4, 'Very Good'), (5, 'Excellent')])
    feedback_comment = models.TextField(blank=True)
    is_corrected = models.BooleanField(default=False)
    corrected_answer = models.TextField(blank=True)
    human_verified = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['timestamp']

# ADD THIS NEW MODEL
class FeedbackData(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    question = models.TextField()
    original_answer = models.TextField()
    corrected_answer = models.TextField(blank=True)
    feedback_type = models.CharField(max_length=20, choices=[
        ('incorrect', 'Incorrect Answer'),
        ('incomplete', 'Incomplete Answer'),
        ('irrelevant', 'Irrelevant Answer'),
        ('good', 'Good Answer')
    ])
    sources_used = models.JSONField(default=list)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
