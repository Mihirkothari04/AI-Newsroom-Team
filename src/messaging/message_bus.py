"""
Message Bus for the AI Newsroom Team messaging system.

This module implements the central message bus that handles routing messages
between agents and maintains a shared memory for the AI Newsroom Team.
"""
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock
from pydantic import BaseModel

from .message import Message, MessageType, MessageStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.messaging")


class MessageBus:
    """
    Central message bus for routing messages between agents.
    
    The MessageBus handles message routing, delivery, and persistence,
    providing a shared memory system for the AI Newsroom Team.
    """
    
    def __init__(self, storage_dir: str = "./data/messages"):
        """
        Initialize the MessageBus.
        
        Args:
            storage_dir: Directory to store message data.
        """
        self.storage_dir = storage_dir
        self.agent_queues: Dict[str, Queue] = {}
        self.agent_callbacks: Dict[str, Callable] = {}
        self.active_tasks: Set[str] = set()
        self.message_history: Dict[str, List[Message]] = {}
        self.lock = Lock()
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info("MessageBus initialized")
    
    def register_agent(self, agent_name: str, callback: Optional[Callable] = None) -> None:
        """
        Register an agent with the message bus.
        
        Args:
            agent_name: Name of the agent.
            callback: Optional callback function to invoke when a message is received.
        """
        with self.lock:
            if agent_name not in self.agent_queues:
                self.agent_queues[agent_name] = Queue()
                if callback:
                    self.agent_callbacks[agent_name] = callback
                logger.info(f"Agent registered: {agent_name}")
            else:
                logger.warning(f"Agent already registered: {agent_name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent from the message bus.
        
        Args:
            agent_name: Name of the agent.
        """
        with self.lock:
            if agent_name in self.agent_queues:
                del self.agent_queues[agent_name]
                if agent_name in self.agent_callbacks:
                    del self.agent_callbacks[agent_name]
                logger.info(f"Agent unregistered: {agent_name}")
            else:
                logger.warning(f"Agent not registered: {agent_name}")
    
    def send_message(self, message: Message) -> bool:
        """
        Send a message to the target agent.
        
        Args:
            message: The message to send.
            
        Returns:
            True if message was queued successfully, False otherwise.
        """
        target_agent = message.target_agent
        
        with self.lock:
            # Check if target agent is registered
            if target_agent not in self.agent_queues:
                logger.error(f"Target agent not registered: {target_agent}")
                return False
            
            # Add to active tasks
            self.active_tasks.add(message.task_id)
            
            # Add to message history
            if message.task_id not in self.message_history:
                self.message_history[message.task_id] = []
            self.message_history[message.task_id].append(message)
            
            # Add to agent's queue
            self.agent_queues[target_agent].put(message)
            
            # Update message history
            message.add_to_history(message.source_agent, "sent")
            
            # Persist message
            self._persist_message(message)
            
            logger.info(f"Message sent from {message.source_agent} to {target_agent} (Type: {message.message_type})")
            
            # Invoke callback if registered
            if target_agent in self.agent_callbacks and self.agent_callbacks[target_agent]:
                try:
                    self.agent_callbacks[target_agent](message)
                except Exception as e:
                    logger.error(f"Error in callback for {target_agent}: {e}")
            
            return True
    
    def receive_message(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message for the specified agent.
        
        Args:
            agent_name: Name of the agent.
            timeout: Optional timeout in seconds.
            
        Returns:
            Message if available, None otherwise.
        """
        if agent_name not in self.agent_queues:
            logger.error(f"Agent not registered: {agent_name}")
            return None
        
        try:
            # Get message from queue
            message = self.agent_queues[agent_name].get(block=True, timeout=timeout)
            
            # Mark as delivered
            message.mark_delivered()
            
            # Update in history
            with self.lock:
                if message.task_id in self.message_history:
                    for i, msg in enumerate(self.message_history[message.task_id]):
                        if msg.message_id == message.message_id:
                            self.message_history[message.task_id][i] = message
                            break
            
            # Persist updated message
            self._persist_message(message)
            
            logger.info(f"Message received by {agent_name} (Type: {message.message_type})")
            
            return message
        except Empty:
            return None
    
    def update_message_status(self, message: Message) -> None:
        """
        Update the status of a message in the message history.
        
        Args:
            message: The updated message.
        """
        with self.lock:
            if message.task_id in self.message_history:
                for i, msg in enumerate(self.message_history[message.task_id]):
                    if msg.message_id == message.message_id:
                        self.message_history[message.task_id][i] = message
                        break
        
        # Persist updated message
        self._persist_message(message)
        
        logger.info(f"Message status updated: {message.message_id} -> {message.status}")
    
    def get_task_messages(self, task_id: str) -> List[Message]:
        """
        Get all messages for a specific task.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            List of messages for the task.
        """
        with self.lock:
            return self.message_history.get(task_id, [])
    
    def get_latest_message(self, task_id: str, message_type: Optional[MessageType] = None) -> Optional[Message]:
        """
        Get the latest message for a task, optionally filtered by type.
        
        Args:
            task_id: ID of the task.
            message_type: Optional type to filter by.
            
        Returns:
            Latest message matching criteria, or None if not found.
        """
        messages = self.get_task_messages(task_id)
        
        if not messages:
            return None
        
        # Filter by type if specified
        if message_type:
            filtered = [m for m in messages if m.message_type == message_type]
            if not filtered:
                return None
            return max(filtered, key=lambda m: m.timestamp)
        
        # Otherwise return latest message
        return max(messages, key=lambda m: m.timestamp)
    
    def _persist_message(self, message: Message) -> None:
        """
        Persist a message to storage.
        
        Args:
            message: The message to persist.
        """
        try:
            # Create task directory if it doesn't exist
            task_dir = os.path.join(self.storage_dir, message.task_id)
            os.makedirs(task_dir, exist_ok=True)
            
            # Save message to file
            file_path = os.path.join(task_dir, f"{message.message_id}.json")
            with open(file_path, 'w') as f:
                # Convert to dict and handle datetime serialization
                message_dict = message.dict()
                message_dict["timestamp"] = message_dict["timestamp"].isoformat()
                json.dump(message_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting message {message.message_id}: {e}")
    
    def load_task_messages(self, task_id: str) -> List[Message]:
        """
        Load all messages for a task from storage.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            List of messages for the task.
        """
        messages = []
        task_dir = os.path.join(self.storage_dir, task_id)
        
        if not os.path.exists(task_dir):
            return messages
        
        try:
            for filename in os.listdir(task_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(task_dir, filename)
                    with open(file_path, 'r') as f:
                        message_dict = json.load(f)
                        # Convert timestamp string back to datetime
                        message_dict["timestamp"] = datetime.fromisoformat(message_dict["timestamp"])
                        messages.append(Message(**message_dict))
        except Exception as e:
            logger.error(f"Error loading messages for task {task_id}: {e}")
        
        return messages
    
    def get_active_tasks(self) -> List[str]:
        """
        Get list of active task IDs.
        
        Returns:
            List of active task IDs.
        """
        with self.lock:
            return list(self.active_tasks)
    
    def mark_task_completed(self, task_id: str) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task.
        """
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
                logger.info(f"Task marked as completed: {task_id}")
    
    def create_shared_memory(self, task_id: str, key: str, value: Any) -> None:
        """
        Store a value in shared memory for a task.
        
        Args:
            task_id: ID of the task.
            key: Key to store the value under.
            value: Value to store.
        """
        # Create a special message to store shared memory
        message = Message(
            task_id=task_id,
            source_agent="SharedMemory",
            target_agent="SharedMemory",
            message_type=MessageType.STATUS_UPDATE,
            payload={"shared_memory_key": key, "shared_memory_value": value}
        )
        
        with self.lock:
            if task_id not in self.message_history:
                self.message_history[task_id] = []
            self.message_history[task_id].append(message)
        
        # Persist shared memory
        self._persist_message(message)
        
        logger.info(f"Shared memory created for task {task_id}: {key}")
    
    def get_shared_memory(self, task_id: str, key: str) -> Optional[Any]:
        """
        Retrieve a value from shared memory for a task.
        
        Args:
            task_id: ID of the task.
            key: Key to retrieve.
            
        Returns:
            Stored value or None if not found.
        """
        messages = self.get_task_messages(task_id)
        
        # Find the latest shared memory message with this key
        shared_memory_messages = [
            m for m in messages 
            if m.source_agent == "SharedMemory" and 
            m.target_agent == "SharedMemory" and
            m.payload.get("shared_memory_key") == key
        ]
        
        if not shared_memory_messages:
            return None
        
        # Get the latest message
        latest = max(shared_memory_messages, key=lambda m: m.timestamp)
        return latest.payload.get("shared_memory_value")
