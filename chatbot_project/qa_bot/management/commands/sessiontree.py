from django.core.management.base import BaseCommand
import json
import sys
import os
import django
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

from qa_bot.models import ChatSession, ChatMessage

class Command(BaseCommand):
    help = 'Session Tree CLI for conversation exploration'

    def add_arguments(self, parser):
        parser.add_argument('action', choices=['tree', 'branch', 'explore'])
        parser.add_argument('--session-id', type=str, help='Session ID')
        parser.add_argument('--thread-id', type=str, help='Thread ID')
        parser.add_argument('--parent-id', type=str, help='Parent node ID for branching')
        parser.add_argument('--message', type=str, help='New message for branching')

    def handle(self, *args, **options):
        action = options['action']
        
        if action == 'tree':
            self.show_tree(options)
        elif action == 'branch':
            self.create_branch(options)
        elif action == 'explore':
            self.explore_session(options)

    def show_tree(self, options):
        session_id = options.get('session_id')
        thread_id = options.get('thread_id')
        
        if session_id:
            self.show_session_tree(session_id)
        elif thread_id:
            self.show_thread_tree(thread_id)
        else:
            self.stdout.write(self.style.ERROR('Session ID or Thread ID required'))

    def show_session_tree(self, session_id):
        try:
            session = ChatSession.objects.get(session_id=session_id)
            messages = session.messages.all().order_by('timestamp')
            
            self.stdout.write(f"🌳 Session Tree: {session_id}")
            self.stdout.write("=" * 60)
            
            for i, msg in enumerate(messages):
                icon = "👤" if msg.role == 'user' else "🤖"
                self.stdout.write(f"{icon} [{i+1}] {msg.role.upper()}")
                self.stdout.write(f"   📝 {msg.content[:100]}...")
                self.stdout.write(f"   🕒 {msg.timestamp}")
                if msg.sources:
                    self.stdout.write(f"   📚 Sources: {len(msg.sources)}")
                self.stdout.write("-" * 40)
                
        except ChatSession.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Session {session_id} not found"))

    def show_thread_tree(self, thread_id):
        from qa_bot.models import SessionTree
        
        try:
            nodes = SessionTree.objects.filter(thread_id=thread_id).order_by('created_at')
            
            if not nodes:
                self.stdout.write(f"🌳 No session tree found for thread: {thread_id}")
                return
                
            self.stdout.write(f"🌳 Session Tree: {thread_id}")
            self.stdout.write("=" * 60)
            
            for node in nodes:
                icon = "❓" if node.node_type == 'question' else "💬"
                self.stdout.write(f"{icon} {node.node_type.upper()}")
                self.stdout.write(f"   Content: {node.content[:80]}...")
                self.stdout.write(f"   Created: {node.created_at}")
                if node.metadata:
                    self.stdout.write(f"   Metadata: {node.metadata}")
                self.stdout.write("-" * 40)
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))

    def create_branch(self, options):
        parent_id = options.get('parent_id')
        message = options.get('message')
        
        if not parent_id or not message:
            self.stdout.write(self.style.ERROR('Parent ID and message required'))
            return
            
        self.stdout.write(f"🌿 Creating branch from: {parent_id}")
        self.stdout.write(f"📝 New message: {message}")
        
        # Simple branching by creating new execution
        from qa_bot.langgraph_timetravel import send_message_with_hitl_timetravel
        import uuid
        
        new_thread_id = f"branch_{uuid.uuid4().hex[:8]}"
        result = send_message_with_hitl_timetravel(message, new_thread_id)
        
        self.stdout.write(self.style.SUCCESS(f"✅ Branch created"))
        self.stdout.write(f"🆔 New Thread: {result['thread_id']}")
        self.stdout.write(f"💬 Answer: {result['answer'][:100]}...")

    def explore_session(self, options):
        session_id = options.get('session_id')
        
        if not session_id:
            self.stdout.write(self.style.ERROR('Session ID required'))
            return
            
        try:
            session = ChatSession.objects.get(session_id=session_id)
            messages = session.messages.all()
            
            self.stdout.write(f"🔍 Exploring Session: {session_id}")
            self.stdout.write(f"📊 Total Messages: {messages.count()}")
            self.stdout.write(f"🕒 Created: {session.created_at}")
            self.stdout.write("=" * 50)
            
            # Show conversation flow
            user_msgs = messages.filter(role='user').count()
            bot_msgs = messages.filter(role='assistant').count()
            
            self.stdout.write(f"👤 User Messages: {user_msgs}")
            self.stdout.write(f"🤖 Bot Messages: {bot_msgs}")
            
            # Show recent interactions
            recent = messages.order_by('-timestamp')[:5]
            self.stdout.write("\n📋 Recent Interactions:")
            for msg in recent:
                icon = "👤" if msg.role == 'user' else "🤖"
                self.stdout.write(f"{icon} {msg.content[:60]}...")
                
        except ChatSession.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Session {session_id} not found"))
