"""
State management module for the AI Newsroom Team orchestrator.

This module provides functionality for tracking and persisting task state,
managing workflow transitions, and handling task history.
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class StateManager:
    """
    Manages state persistence and retrieval for the orchestrator.
    
    This class handles saving task state to disk, loading state from disk,
    and maintaining an in-memory state cache for active tasks.
    """
    
    def __init__(self, state_dir: str = "./data/state"):
        """
        Initialize the StateManager.
        
        Args:
            state_dir: Directory to store state files.
        """
        self.state_dir = state_dir
        self.state_cache = {}
        
        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)
    
    def save_task_state(self, task_id: str, state: Dict[str, Any]) -> None:
        """
        Save task state to disk and update cache.
        
        Args:
            task_id: Unique identifier for the task.
            state: Task state to save.
        """
        # Update cache
        self.state_cache[task_id] = state
        
        # Save to disk
        file_path = os.path.join(self.state_dir, f"{task_id}.json")
        with open(file_path, 'w') as f:
            json.dump(state, f, default=self._json_serializer)
    
    def load_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Load task state from cache or disk.
        
        Args:
            task_id: Unique identifier for the task.
            
        Returns:
            Task state if found, None otherwise.
        """
        # Check cache first
        if task_id in self.state_cache:
            return self.state_cache[task_id]
        
        # Try to load from disk
        file_path = os.path.join(self.state_dir, f"{task_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                state = json.load(f)
                self.state_cache[task_id] = state
                return state
        
        return None
    
    def list_tasks(self) -> List[str]:
        """
        List all task IDs with saved state.
        
        Returns:
            List of task IDs.
        """
        tasks = []
        for filename in os.listdir(self.state_dir):
            if filename.endswith('.json'):
                tasks.append(filename[:-5])  # Remove .json extension
        return tasks
    
    def delete_task_state(self, task_id: str) -> bool:
        """
        Delete task state from disk and cache.
        
        Args:
            task_id: Unique identifier for the task.
            
        Returns:
            True if state was deleted, False otherwise.
        """
        # Remove from cache
        if task_id in self.state_cache:
            del self.state_cache[task_id]
        
        # Remove from disk
        file_path = os.path.join(self.state_dir, f"{task_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        
        return False
    
    def _json_serializer(self, obj):
        """
        Custom JSON serializer for objects not serializable by default json code.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            Serialized representation of the object.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        raise TypeError(f"Type {type(obj)} not serializable")


class WorkflowManager:
    """
    Manages workflow definitions and transitions.
    
    This class handles loading workflow definitions, determining valid transitions,
    and managing workflow state for tasks.
    """
    
    def __init__(self, workflow_dir: str = "./config/workflows"):
        """
        Initialize the WorkflowManager.
        
        Args:
            workflow_dir: Directory containing workflow definition files.
        """
        self.workflow_dir = workflow_dir
        self.workflows = self._load_workflows()
    
    def _load_workflows(self) -> Dict[str, Dict[str, Any]]:
        """
        Load workflow definitions from files.
        
        Returns:
            Dictionary mapping workflow names to their definitions.
        """
        # Default workflows (in case files don't exist)
        default_workflows = {
            "standard_news": {
                "stages": [
                    {"name": "GATHERING", "agent": "GatherBot"},
                    {"name": "WRITING", "agent": "WriterBot"},
                    {"name": "FACT_CHECKING", "agent": "FactCheckBot"},
                    {"name": "COMPLETED", "agent": None}
                ],
                "max_revision_cycles": 2
            },
            "breaking_news": {
                "stages": [
                    {"name": "GATHERING", "agent": "GatherBot"},
                    {"name": "WRITING", "agent": "WriterBot"},
                    {"name": "FACT_CHECKING", "agent": "FactCheckBot"},
                    {"name": "COMPLETED", "agent": None}
                ],
                "max_revision_cycles": 1
            },
            "feature_article": {
                "stages": [
                    {"name": "GATHERING", "agent": "GatherBot"},
                    {"name": "WRITING", "agent": "WriterBot"},
                    {"name": "FACT_CHECKING", "agent": "FactCheckBot"},
                    {"name": "EDITING", "agent": "EditorBot"},
                    {"name": "COMPLETED", "agent": None}
                ],
                "max_revision_cycles": 3
            }
        }
        
        # Try to load from files if directory exists
        if os.path.exists(self.workflow_dir):
            for filename in os.listdir(self.workflow_dir):
                if filename.endswith('.json'):
                    workflow_name = filename[:-5]  # Remove .json extension
                    file_path = os.path.join(self.workflow_dir, filename)
                    with open(file_path, 'r') as f:
                        default_workflows[workflow_name] = json.load(f)
        
        return default_workflows
    
    def get_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """
        Get a workflow definition by name.
        
        Args:
            workflow_name: Name of the workflow.
            
        Returns:
            Workflow definition.
        """
        return self.workflows.get(workflow_name, self.workflows["standard_news"])
    
    def get_next_stage(self, workflow_name: str, current_stage: str) -> Optional[Dict[str, Any]]:
        """
        Get the next stage in a workflow.
        
        Args:
            workflow_name: Name of the workflow.
            current_stage: Current stage name.
            
        Returns:
            Next stage definition or None if at the end.
        """
        workflow = self.get_workflow(workflow_name)
        
        # Find current stage index
        current_index = -1
        for i, stage in enumerate(workflow["stages"]):
            if stage["name"] == current_stage:
                current_index = i
                break
        
        # If found and not at the end, return next stage
        if current_index >= 0 and current_index < len(workflow["stages"]) - 1:
            return workflow["stages"][current_index + 1]
        
        return None
    
    def get_stage_for_revision(self, workflow_name: str, feedback_type: str) -> Optional[Dict[str, Any]]:
        """
        Determine which stage to return to based on feedback.
        
        Args:
            workflow_name: Name of the workflow.
            feedback_type: Type of feedback requiring revision.
            
        Returns:
            Stage to return to or None if no revision needed.
        """
        workflow = self.get_workflow(workflow_name)
        
        # Map feedback types to appropriate stages
        # This is a simplified example - real logic would be more sophisticated
        feedback_stage_map = {
            "content_issue": "WRITING",
            "fact_error": "GATHERING",
            "style_issue": "WRITING",
            "structure_issue": "WRITING"
        }
        
        target_stage_name = feedback_stage_map.get(feedback_type)
        if not target_stage_name:
            return None
        
        # Find the stage in the workflow
        for stage in workflow["stages"]:
            if stage["name"] == target_stage_name:
                return stage
        
        return None
