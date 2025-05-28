"""
Integration tests for the AI Newsroom Team.

This module provides integration tests for the interaction between agents
and the messaging system.
"""
import unittest
import os
import json
import time
from unittest.mock import MagicMock, patch

from src.orchestrator import NewsroomOrchestrator
from src.agents.gatherbot import GatherBot
from src.agents.writerbot import WriterBot
from src.agents.factcheckbot import FactCheckBot
from src.messaging import MessageBus, Message, MessageType


class TestAgentIntegration(unittest.TestCase):
    """Test cases for agent integration with the messaging system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test directory for message storage
        os.makedirs("./test_data/messages", exist_ok=True)
        
        # Initialize message bus with test storage
        self.message_bus = MessageBus(storage_dir="./test_data/messages")
        
        # Initialize agents
        self.gather_bot = GatherBot(name="TestGatherBot", message_bus=self.message_bus)
        self.writer_bot = WriterBot(name="TestWriterBot", message_bus=self.message_bus)
        self.fact_check_bot = FactCheckBot(name="TestFactCheckBot", message_bus=self.message_bus)
        
        # Initialize orchestrator
        self.orchestrator = NewsroomOrchestrator(message_bus=self.message_bus)
    
    def test_message_passing(self):
        """Test message passing between agents."""
        # Create a test task
        task_id = "test_task_001"
        
        # Create a test message
        test_payload = {
            "topic": "test topic",
            "sources": ["news_api"],
            "max_items": 3
        }
        
        # Send message from orchestrator to GatherBot
        sent = self.message_bus.send_message(Message(
            task_id=task_id,
            source_agent="Orchestrator",
            target_agent="TestGatherBot",
            message_type=MessageType.TASK_ASSIGNMENT,
            payload=test_payload
        ))
        
        self.assertTrue(sent, "Message should be sent successfully")
        
        # Check if message is in the task history
        task_messages = self.message_bus.get_task_messages(task_id)
        self.assertEqual(len(task_messages), 1, "Task should have one message")
        
        # Check message content
        message = task_messages[0]
        self.assertEqual(message.source_agent, "Orchestrator")
        self.assertEqual(message.target_agent, "TestGatherBot")
        self.assertEqual(message.message_type, MessageType.TASK_ASSIGNMENT)
        self.assertEqual(message.payload["topic"], "test topic")
    
    def test_shared_memory(self):
        """Test shared memory functionality."""
        # Create a test task
        task_id = "test_task_002"
        
        # Store a value in shared memory
        self.message_bus.create_shared_memory(task_id, "test_key", "test_value")
        
        # Retrieve the value
        value = self.message_bus.get_shared_memory(task_id, "test_key")
        
        self.assertEqual(value, "test_value", "Shared memory should store and retrieve values")
    
    @patch('src.agents.gatherbot.GatherBot.process_task')
    def test_agent_response(self, mock_process_task):
        """Test agent response to messages."""
        # Mock the process_task method
        mock_process_task.return_value = {
            "topic": "test topic",
            "items": [{"title": "Test Item", "content": "Test content"}],
            "sources_used": ["Test Source"]
        }
        
        # Create a test task
        task_id = "test_task_003"
        
        # Register a message handler for GatherBot
        self.gather_bot.register_handler(
            MessageType.TASK_ASSIGNMENT,
            lambda msg: self.gather_bot.send_message(
                target_agent="Orchestrator",
                message_type=MessageType.GATHER_RESULT,
                task_id=msg.task_id,
                payload=self.gather_bot.process_task(msg.payload)
            )
        )
        
        # Send message from orchestrator to GatherBot
        self.message_bus.send_message(Message(
            task_id=task_id,
            source_agent="Orchestrator",
            target_agent="TestGatherBot",
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"topic": "test topic"}
        ))
        
        # Manually trigger the message handler (in a real scenario this would be automatic)
        message = self.message_bus.receive_message("TestGatherBot")
        if message:
            handler = self.gather_bot.message_handlers.get(message.message_type)
            if handler:
                handler(message)
        
        # Check for response message
        task_messages = self.message_bus.get_task_messages(task_id)
        self.assertGreaterEqual(len(task_messages), 2, "Task should have at least two messages")
        
        # Find the response message
        response_message = None
        for msg in task_messages:
            if msg.message_type == MessageType.GATHER_RESULT:
                response_message = msg
                break
        
        self.assertIsNotNone(response_message, "Response message should exist")
        if response_message:
            self.assertEqual(response_message.source_agent, "TestGatherBot")
            self.assertEqual(response_message.target_agent, "Orchestrator")
            self.assertIn("items", response_message.payload)


class TestOrchestratorWorkflow(unittest.TestCase):
    """Test cases for the orchestrator workflow management."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test directory for message storage
        os.makedirs("./test_data/messages", exist_ok=True)
        
        # Initialize message bus with test storage
        self.message_bus = MessageBus(storage_dir="./test_data/messages")
        
        # Initialize orchestrator
        self.orchestrator = NewsroomOrchestrator(message_bus=self.message_bus)
    
    def test_task_creation(self):
        """Test task creation and status tracking."""
        # Create a test task
        task_id = self.orchestrator.create_task(
            topic="test topic",
            task_type="news_article",
            priority="high"
        )
        
        self.assertIsNotNone(task_id, "Task ID should be generated")
        
        # Check task status
        status = self.orchestrator.get_task_status(task_id)
        
        self.assertEqual(status["status"], "pending", "New task should have pending status")
        self.assertEqual(status["topic"], "test topic", "Task should have correct topic")
        self.assertEqual(status["priority"], "high", "Task should have correct priority")
    
    @patch('src.orchestrator.orchestrator.NewsroomOrchestrator._assign_gather_task')
    def test_workflow_start(self, mock_assign_gather):
        """Test workflow initiation."""
        # Create a test task
        task_id = self.orchestrator.create_task(
            topic="test topic",
            task_type="news_article"
        )
        
        # Start the workflow
        self.orchestrator.start_task(task_id)
        
        # Verify that gather task was assigned
        mock_assign_gather.assert_called_once()
        
        # Check task status
        status = self.orchestrator.get_task_status(task_id)
        self.assertEqual(status["status"], "in_progress", "Task should be in progress")


if __name__ == '__main__':
    unittest.main()
