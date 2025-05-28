"""
Agent integration module for the AI Newsroom Team.

This module provides utilities for integrating agents with the messaging system
and facilitating communication between agents.
"""
import logging
from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod

from src.messaging import MessageBus, Message, MessageType, MessageStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.agent_integration")


class Agent(ABC):
    """
    Abstract base class for all agents in the AI Newsroom Team.
    
    This class defines the common interface that all agents must implement
    to interact with the messaging system.
    """
    
    def __init__(self, name: str, message_bus: MessageBus):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent.
            message_bus: MessageBus instance for communication.
        """
        self.name = name
        self.message_bus = message_bus
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Register with message bus
        self.message_bus.register_agent(self.name, self._on_message_received)
        
        logger.info(f"Agent {self.name} initialized")
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle.
            handler: Function to call when message is received.
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Agent {self.name} registered handler for {message_type}")
    
    def send_message(self, target_agent: str, message_type: MessageType, 
                    task_id: str, payload: Dict[str, Any],
                    instructions: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to another agent.
        
        Args:
            target_agent: Name of the target agent.
            message_type: Type of message.
            task_id: ID of the task.
            payload: Message payload.
            instructions: Optional instructions for the target agent.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        message = Message(
            source_agent=self.name,
            target_agent=target_agent,
            message_type=message_type,
            task_id=task_id,
            payload=payload,
            instructions=instructions or {}
        )
        
        return self.message_bus.send_message(message)
    
    def _on_message_received(self, message: Message) -> None:
        """
        Callback for when a message is received.
        
        Args:
            message: The received message.
        """
        logger.info(f"Agent {self.name} received message of type {message.message_type}")
        
        # Mark message as processing
        message.mark_processing()
        self.message_bus.update_message_status(message)
        
        # Handle message based on type
        if message.message_type in self.message_handlers:
            try:
                self.message_handlers[message.message_type](message)
                message.mark_completed()
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
                message.mark_failed(str(e))
        else:
            logger.warning(f"No handler registered for message type {message.message_type}")
            message.mark_failed(f"No handler for message type {message.message_type}")
        
        # Update message status
        self.message_bus.update_message_status(message)
    
    @abstractmethod
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        
        Args:
            task_data: Task data to process.
            
        Returns:
            Result of processing.
        """
        pass


class AgentFactory:
    """
    Factory class for creating agent instances.
    
    This class manages the creation and registration of agents with the
    messaging system.
    """
    
    def __init__(self, message_bus: MessageBus):
        """
        Initialize the agent factory.
        
        Args:
            message_bus: MessageBus instance for communication.
        """
        self.message_bus = message_bus
        self.agent_classes: Dict[str, Type[Agent]] = {}
        
        logger.info("AgentFactory initialized")
    
    def register_agent_class(self, agent_type: str, agent_class: Type[Agent]) -> None:
        """
        Register an agent class with the factory.
        
        Args:
            agent_type: Type identifier for the agent.
            agent_class: Agent class to register.
        """
        self.agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class: {agent_type}")
    
    def create_agent(self, agent_type: str, name: str, **kwargs) -> Optional[Agent]:
        """
        Create an agent instance.
        
        Args:
            agent_type: Type of agent to create.
            name: Name for the agent instance.
            **kwargs: Additional arguments to pass to the agent constructor.
            
        Returns:
            Agent instance or None if agent_type is not registered.
        """
        if agent_type not in self.agent_classes:
            logger.error(f"Unknown agent type: {agent_type}")
            return None
        
        # Create agent instance
        agent_class = self.agent_classes[agent_type]
        agent = agent_class(name=name, message_bus=self.message_bus, **kwargs)
        
        logger.info(f"Created agent: {name} (type: {agent_type})")
        return agent
