from typing import Dict, List, Optional, Any
from .models import SessionTree
import uuid

class SessionTreeManager:
    """Manage session history as tree structure"""
    
    def create_session_node(self, thread_id: str, content: str, node_type: str, parent_id: str = None, metadata: Dict = None) -> str:
        """Create new node in session tree"""
        parent_node = None
        if parent_id:
            try:
                parent_node = SessionTree.objects.get(session_id=parent_id)
            except SessionTree.DoesNotExist:
                pass
        
        node = SessionTree.objects.create(
            thread_id=thread_id,
            parent_node=parent_node,
            node_type=node_type,
            content=content,
            metadata=metadata or {}
        )
        
        return str(node.session_id)
    
    def get_session_tree(self, thread_id: str) -> Dict:
        """Get complete session tree"""
        try:
            nodes = SessionTree.objects.filter(thread_id=thread_id).order_by('created_at')
            
            tree = []
            for node in nodes:
                tree.append({
                    'id': str(node.session_id),
                    'parent_id': str(node.parent_node.session_id) if node.parent_node else None,
                    'type': node.node_type,
                    'content': node.content,
                    'metadata': node.metadata,
                    'created_at': node.created_at.isoformat()
                })
            
            return {'tree': tree, 'thread_id': thread_id}
        except Exception as e:
            return {'tree': [], 'error': str(e)}
    
    def create_branch(self, parent_node_id: str, new_content: str, metadata: Dict = None) -> str:
        """Create new branch from existing node"""
        try:
            parent = SessionTree.objects.get(session_id=parent_node_id)
            new_thread_id = f"{parent.thread_id}_branch_{uuid.uuid4().hex[:8]}"
            
            return self.create_session_node(
                thread_id=new_thread_id,
                content=new_content,
                node_type='question',
                parent_id=parent_node_id,
                metadata=metadata
            )
        except Exception as e:
            return None
    
    def get_conversation_path(self, node_id: str) -> List[Dict]:
        """Get conversation path from root to node"""
        try:
            node = SessionTree.objects.get(session_id=node_id)
            path = []
            current = node
            
            while current:
                path.insert(0, {
                    'id': str(current.session_id),
                    'type': current.node_type,
                    'content': current.content,
                    'created_at': current.created_at.isoformat()
                })
                current = current.parent_node
            
            return path
        except Exception as e:
            return []
