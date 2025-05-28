"""
Message class for the AI Newsroom Team messaging system.

This module defines the message structure and types used for communication
between agents in the AI Newsroom Team.
"""
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Enum representing the types of messages that can be passed between agents."""
    TASK_ASSIGNMENT = "task_assignment"
    GATHER_RESULT = "gather_result"
    ARTICLE_DRAFT = "article_draft"
    VERIFICATION_RESULT = "verification_result"
    REVISION_REQUEST = "revision_request"
    FINAL_ARTICLE = "final_article"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    COMMAND = "command"


class MessageStatus(str, Enum):
    """Enum representing the possible statuses of a message."""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Message(BaseModel):
    """
    Model representing a message passed between agents in the AI Newsroom Team.
    
    Messages are the primary means of communication between agents, containing
    both metadata about the communication and the actual payload data.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    source_agent: str
    target_agent: str
    message_type: MessageType
    status: MessageStatus = MessageStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: Dict[str, Any] = Field(default_factory=dict)
    instructions: Optional[Dict[str, Any]] = Field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_to_history(self, agent: str, action: str) -> None:
        """
        Add an entry to the message history.
        
        Args:
            agent: The agent performing the action.
            action: The action performed.
        """
        self.history.append({
            "agent": agent,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
    
    def mark_delivered(self) -> None:
        """Mark the message as delivered to the target agent."""
        self.status = MessageStatus.DELIVERED
        self.add_to_history(self.target_agent, "received")
    
    def mark_processing(self) -> None:
        """Mark the message as being processed by the target agent."""
        self.status = MessageStatus.PROCESSING
        self.add_to_history(self.target_agent, "processing")
    
    def mark_completed(self) -> None:
        """Mark the message as completed by the target agent."""
        self.status = MessageStatus.COMPLETED
        self.add_to_history(self.target_agent, "completed")
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark the message as failed.
        
        Args:
            error_message: Description of the failure.
        """
        self.status = MessageStatus.FAILED
        self.add_to_history(self.target_agent, f"failed: {error_message}")
    
    def create_response(self, message_type: MessageType, payload: Dict[str, Any]) -> 'Message':
        """
        Create a response message to this message.
        
        Args:
            message_type: Type of the response message.
            payload: Data payload for the response.
            
        Returns:
            New Message object representing the response.
        """
        return Message(
            task_id=self.task_id,
            source_agent=self.target_agent,
            target_agent=self.source_agent,
            message_type=message_type,
            payload=payload,
            history=[{
                "agent": self.target_agent,
                "action": "responding",
                "timestamp": datetime.now().isoformat(),
                "in_response_to": self.message_id
            }]
        )
