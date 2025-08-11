from django.core.management.base import BaseCommand
import json
import sys
import os
import django
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

from qa_bot.langgraph_timetravel import send_message_with_hitl_timetravel, get_thread_history, rewind_to_checkpoint

class Command(BaseCommand):
    help = 'Time Travel CLI for pipeline execution'

    def add_arguments(self, parser):
        parser.add_argument('action', choices=['execute', 'rerun', 'history', 'rewind'])
        parser.add_argument('--thread-id', type=str, help='Thread ID')
        parser.add_argument('--message', type=str, help='Message to process')
        parser.add_argument('--nodes', type=str, help='Comma-separated node names to rerun')
        parser.add_argument('--checkpoint', type=str, help='Checkpoint ID for rewind')
        parser.add_argument('--context', type=str, help='Context choice to avoid ambiguity')

    def handle(self, *args, **options):
        action = options['action']
        
        if action == 'execute':
            self.execute_pipeline(options)
        elif action == 'rerun':
            self.rerun_nodes(options)
        elif action == 'history':
            self.show_history(options)
        elif action == 'rewind':
            self.rewind_execution(options)

    def execute_pipeline(self, options):
        message = options.get('message')
        thread_id = options.get('thread_id')
        context_choice = options.get('context')
        
        if not message:
            self.stdout.write(self.style.ERROR('Message required'))
            return
            
        self.stdout.write(f"🚀 Executing: {message}")
        if thread_id:
            self.stdout.write(f"📍 Thread: {thread_id}")
        
        # Use time travel engine instead of regular function
        from qa_bot.langgraph_timetravel import execute_with_time_travel
        
        try:
            result = execute_with_time_travel(
                message=message,
                thread_id=thread_id,
                context_choice=context_choice
            )
            
            self.stdout.write(self.style.SUCCESS(f"✅ Answer: {result['answer']}"))
            self.stdout.write(f"🔗 Thread ID: {result['thread_id']}")
            self.stdout.write(f"📊 Confidence: {result['confidence_score']}")
            if result.get('sources'):
                self.stdout.write(f"📚 Sources: {', '.join(result['sources'])}")
            if result.get('execution_log'):
                self.stdout.write(f"⚡ Execution: {', '.join(result['execution_log'])}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))


    def rerun_nodes(self, options):
        thread_id = options.get('thread_id')
        nodes = options.get('nodes', '').split(',')
        message = options.get('message', 'Rerun execution')
        
        if not thread_id:
            self.stdout.write(self.style.ERROR('Thread ID required'))
            return
            
        self.stdout.write(f"🔄 Re-running nodes: {', '.join(nodes)}")
        self.stdout.write(f"📍 Thread: {thread_id}")
        
        # Simple rerun by executing new message
        result = send_message_with_hitl_timetravel(message, thread_id)
        
        self.stdout.write(self.style.SUCCESS(f"✅ Re-executed"))
        self.stdout.write(f"📝 Answer: {result['answer']}")
        self.stdout.write(f"📊 Confidence: {result['confidence_score']}")


    def show_history(self, options):
        thread_id = options.get('thread_id')
        
        if not thread_id:
            self.stdout.write(self.style.ERROR('Thread ID required'))
            return
        
        # Check database for execution records
        from qa_bot.models import NodeExecution
        
        try:
            executions = NodeExecution.objects.filter(thread_id=thread_id).order_by('executed_at')
            
            if not executions:
                self.stdout.write(f"📜 No execution history found for thread: {thread_id}")
                return
                
            self.stdout.write(f"📜 Execution History for thread: {thread_id}")
            self.stdout.write("=" * 50)
            
            for exec in executions:
                self.stdout.write(f"🔹 Node: {exec.node_name}")
                self.stdout.write(f"   Runtime: {exec.runtime_ms}ms")
                self.stdout.write(f"   Input: {str(exec.input_data)[:100]}...")
                self.stdout.write(f"   Output: {str(exec.output_data)[:100]}...")
                self.stdout.write(f"   Executed: {exec.executed_at}")
                self.stdout.write("-" * 30)
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            

    def rewind_execution(self, options):
        thread_id = options.get('thread_id')
        checkpoint_id = options.get('checkpoint')
        
        if not thread_id or not checkpoint_id:
            self.stdout.write(self.style.ERROR('Thread ID and Checkpoint ID required'))
            return
            
        result = rewind_to_checkpoint(thread_id, checkpoint_id)
        
        if result.get('success'):
            self.stdout.write(self.style.SUCCESS(f"⏪ Rewound to checkpoint: {checkpoint_id}"))
            self.stdout.write(f"📍 Thread: {thread_id}")
        else:
            self.stdout.write(self.style.ERROR(f"Failed: {result.get('error')}"))
