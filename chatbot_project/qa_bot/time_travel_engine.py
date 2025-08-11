import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from .models import NodeExecution, ExecutionCheckpoint, SessionTree
import uuid

class TimeTravelEngine:
    """Core time travel execution engine"""
    
    def __init__(self):
        self.node_cache = {}
        self.execution_graph = {}
        
    def generate_cache_key(self, node_name: str, input_data: Dict) -> str:
        """Generate cache key for node execution"""
        content = f"{node_name}:{json.dumps(input_data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_downstream_nodes(self, target_node: str, all_nodes: List[str]) -> Set[str]:
        """Get all nodes that depend on target_node"""
        downstream = set()
        
        # Simple dependency mapping for chatbot pipeline
        dependencies = {
            'retrieve': [],
            'hitl_decision': ['retrieve'],
            'generate': ['retrieve', 'hitl_decision']
        }
        
        for node, deps in dependencies.items():
            if target_node in deps:
                downstream.add(node)
                downstream.update(self.get_downstream_nodes(node, all_nodes))
        
        return downstream
    
    def should_reexecute_node(self, node_name: str, input_data: Dict, thread_id: str, force_nodes: Set[str]) -> bool:
        """Determine if node should be re-executed"""
        if node_name in force_nodes:
            return True
            
        cache_key = self.generate_cache_key(node_name, input_data)
        
        # Check if cached result exists
        try:
            cached = NodeExecution.objects.filter(
                thread_id=thread_id,
                node_name=node_name,
                cache_key=cache_key
            ).first()
            return cached is None
        except:
            return True
    
    def execute_node_with_timing(self, node_name: str, node_func, input_data: Dict, thread_id: str) -> Dict:
        """Execute node with timing and caching"""
        start_time = time.time()
        
        try:
            result = node_func(input_data)
            runtime_ms = int((time.time() - start_time) * 1000)
            
            # Store execution record
            cache_key = self.generate_cache_key(node_name, input_data)
            NodeExecution.objects.create(
                thread_id=thread_id,
                node_name=node_name,
                input_data=input_data,
                output_data=result,
                runtime_ms=runtime_ms,
                cache_key=cache_key
            )
            
            return {**result, '_runtime_ms': runtime_ms, '_executed': True}
        except Exception as e:
            return {'error': str(e), '_runtime_ms': int((time.time() - start_time) * 1000), '_executed': True}
    
    def get_cached_result(self, node_name: str, input_data: Dict, thread_id: str) -> Optional[Dict]:
        """Get cached result for node"""
        cache_key = self.generate_cache_key(node_name, input_data)
        
        try:
            cached = NodeExecution.objects.filter(
                thread_id=thread_id,
                node_name=node_name,
                cache_key=cache_key
            ).first()
            
            if cached:
                return {**cached.output_data, '_runtime_ms': 0, '_cached': True}
        except:
            pass
        return None
    
    def time_travel_execute(self, pipeline_nodes: Dict, initial_state: Dict, thread_id: str, rerun_nodes: Set[str] = None) -> Dict:
        """Execute pipeline with time travel support"""
        if rerun_nodes is None:
            rerun_nodes = set()
        
        # Get all nodes that need re-execution
        nodes_to_rerun = set(rerun_nodes)
        for node in rerun_nodes:
            nodes_to_rerun.update(self.get_downstream_nodes(node, list(pipeline_nodes.keys())))
        
        state = initial_state.copy()
        execution_log = []
        
        # Execute nodes in order
        for node_name, node_func in pipeline_nodes.items():
            if self.should_reexecute_node(node_name, state, thread_id, nodes_to_rerun):
                result = self.execute_node_with_timing(node_name, node_func, state, thread_id)
                execution_log.append(f"✓ {node_name}: {result.get('_runtime_ms', 0)}ms")
            else:
                result = self.get_cached_result(node_name, state, thread_id)
                if result:
                    execution_log.append(f"⚡ {node_name}: cached")
                else:
                    result = self.execute_node_with_timing(node_name, node_func, state, thread_id)
                    execution_log.append(f"✓ {node_name}: {result.get('_runtime_ms', 0)}ms")
            
            state.update(result)
        
        state['_execution_log'] = execution_log
        return state
    
    def create_checkpoint(self, thread_id: str, state: Dict) -> str:
        """Create execution checkpoint"""
        checkpoint = ExecutionCheckpoint.objects.create(
            thread_id=thread_id,
            state_data=state
        )
        return str(checkpoint.checkpoint_id)
