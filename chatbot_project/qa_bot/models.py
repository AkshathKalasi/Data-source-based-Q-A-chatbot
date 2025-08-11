from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

class ChatSession(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_awaiting_source = models.BooleanField(default=False)
    last_question = models.TextField(blank=True)

    @property
    def awaiting_source(self):
        return self.is_awaiting_source
    
    @awaiting_source.setter
    def awaiting_source(self, value):
        self.is_awaiting_source = value

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
    
    confidence_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    feedback_rating = models.IntegerField(
        null=True, blank=True, 
        choices=[(1, 'Poor'), (2, 'Fair'), (3, 'Good'), (4, 'Very Good'), (5, 'Excellent')]
    )
    feedback_comment = models.TextField(blank=True)
    is_corrected = models.BooleanField(default=False)
    corrected_answer = models.TextField(blank=True)
    human_verified = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['timestamp']

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
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

class QAWorkflow(models.Model):
    STATUS_CHOICES = [
        ('generated', 'Generated'),
        ('pending_review', 'Pending Human Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('refined', 'Refined'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='qa_workflows')
    user_question = models.TextField()
    generated_answer = models.TextField()
    refined_answer = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='generated')
    human_feedback = models.TextField(blank=True)
    chat_context = models.JSONField(default=list)
    sources = models.JSONField(default=list)
    confidence_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class HumanReview(models.Model):
    DECISION_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    
    qa_workflow = models.OneToOneField(QAWorkflow, on_delete=models.CASCADE, related_name='review')
    reviewer_notes = models.TextField(blank=True)
    decision = models.CharField(max_length=10, choices=DECISION_CHOICES, default='pending')
    reviewed_at = models.DateTimeField(null=True, blank=True)


# Add these models to the end of your models.py file:

# ==================== TIME TRAVEL & SESSION TREE MODELS ====================

class SessionTree(models.Model):
    """Tree structure for session history"""
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    thread_id = models.CharField(max_length=100, db_index=True)
    parent_node = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='children')
    node_type = models.CharField(max_length=20, choices=[('question', 'Question'), ('response', 'Response')])
    content = models.TextField()
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['thread_id', 'created_at'])]

class NodeExecution(models.Model):
    """Track individual node executions for time travel"""
    execution_id = models.UUIDField(default=uuid.uuid4, unique=True)
    thread_id = models.CharField(max_length=100, db_index=True)
    node_name = models.CharField(max_length=50)
    input_data = models.JSONField(default=dict)
    output_data = models.JSONField(default=dict)
    runtime_ms = models.IntegerField(default=0)
    dependencies = models.JSONField(default=list)  # List of node names this depends on
    cache_key = models.CharField(max_length=255, db_index=True)
    executed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['thread_id', 'executed_at'])]

class ExecutionCheckpoint(models.Model):
    """Store execution states for time travel"""
    checkpoint_id = models.UUIDField(default=uuid.uuid4, unique=True)
    thread_id = models.CharField(max_length=100, db_index=True)
    state_data = models.JSONField(default=dict)
    node_executions = models.ManyToManyField(NodeExecution, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [models.Index(fields=['thread_id', 'created_at'])]
