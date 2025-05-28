"""
End-to-end workflow test for the AI Newsroom Team.

This module provides a test script to validate the complete multi-agent workflow,
from news gathering to article writing and fact-checking.
"""
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.orchestrator import NewsroomOrchestrator
from src.agents.gatherbot import GatherBot
from src.agents.writerbot import WriterBot
from src.agents.factcheckbot import FactCheckBot
from src.messaging import MessageBus, MessageType
from src.tools import NewsAPIClient, WebScraper, LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("workflow_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("newsroom.test")

def setup_test_environment():
    """Set up the test environment with necessary directories."""
    os.makedirs("./data/messages", exist_ok=True)
    os.makedirs("./data/articles", exist_ok=True)
    os.makedirs("./data/test_results", exist_ok=True)
    
    logger.info("Test environment set up")

def create_test_agents():
    """Create and configure all agents for testing."""
    # Initialize message bus
    message_bus = MessageBus(storage_dir="./data/messages")
    
    # Initialize orchestrator
    orchestrator = NewsroomOrchestrator(message_bus=message_bus)
    
    # Initialize agents
    gather_bot = GatherBot(name="GatherBot", message_bus=message_bus)
    writer_bot = WriterBot(name="WriterBot", message_bus=message_bus)
    fact_check_bot = FactCheckBot(name="FactCheckBot", message_bus=message_bus)
    
    logger.info("Test agents created")
    
    return {
        "message_bus": message_bus,
        "orchestrator": orchestrator,
        "gather_bot": gather_bot,
        "writer_bot": writer_bot,
        "fact_check_bot": fact_check_bot
    }

def run_test_workflow(agents, topic="artificial intelligence"):
    """
    Run a complete end-to-end workflow test.
    
    Args:
        agents: Dictionary of agent instances.
        topic: Topic to generate news about.
    """
    logger.info(f"Starting workflow test for topic: {topic}")
    
    orchestrator = agents["orchestrator"]
    
    # Start a new task
    task_id = orchestrator.create_task(
        topic=topic,
        task_type="news_article",
        priority="high"
    )
    
    logger.info(f"Created task: {task_id}")
    
    # Start the workflow
    orchestrator.start_task(task_id)
    
    # Wait for task to complete
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        task_status = orchestrator.get_task_status(task_id)
        
        if task_status["status"] == "completed":
            logger.info(f"Task completed successfully in {time.time() - start_time:.2f} seconds")
            break
        elif task_status["status"] == "failed":
            logger.error(f"Task failed: {task_status.get('error', 'Unknown error')}")
            break
        
        logger.info(f"Task status: {task_status['status']} - {task_status.get('current_step', 'Unknown step')}")
        time.sleep(5)  # Check every 5 seconds
    else:
        logger.warning(f"Task did not complete within {max_wait_time} seconds")
    
    # Get final results
    results = orchestrator.get_task_results(task_id)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"./data/test_results/workflow_test_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {result_file}")
    
    return results

def validate_results(results):
    """
    Validate the workflow results.
    
    Args:
        results: Results from the workflow test.
        
    Returns:
        Dictionary with validation results.
    """
    validation = {
        "success": False,
        "issues": [],
        "metrics": {}
    }
    
    # Check if results exist
    if not results:
        validation["issues"].append("No results returned")
        return validation
    
    # Check for article content
    if "article" not in results:
        validation["issues"].append("No article in results")
    elif not results["article"].get("content"):
        validation["issues"].append("Article has no content")
    else:
        # Calculate metrics
        content = results["article"]["content"]
        validation["metrics"]["word_count"] = len(content.split())
        validation["metrics"]["paragraph_count"] = len([p for p in content.split('\n\n') if p.strip()])
    
    # Check for verification results
    if "verification" not in results:
        validation["issues"].append("No verification results")
    else:
        verification = results["verification"]
        validation["metrics"]["accuracy_score"] = verification.get("overall_accuracy_score", 0)
        validation["metrics"]["verified_claims"] = len(verification.get("verified_claims", []))
        validation["metrics"]["unverified_claims"] = len(verification.get("unverified_claims", []))
    
    # Check for sources
    if "sources" not in results or not results["sources"]:
        validation["issues"].append("No sources in results")
    else:
        validation["metrics"]["source_count"] = len(results["sources"])
    
    # Overall success determination
    if not validation["issues"] and validation["metrics"].get("accuracy_score", 0) >= 0.7:
        validation["success"] = True
    
    return validation

def main():
    """Run the complete workflow test."""
    logger.info("Starting AI Newsroom Team workflow test")
    
    # Set up test environment
    setup_test_environment()
    
    # Create agents
    agents = create_test_agents()
    
    # Test topics
    test_topics = [
        "artificial intelligence",
        "climate change",
        "space exploration"
    ]
    
    # Run tests for each topic
    all_validations = {}
    
    for topic in test_topics:
        logger.info(f"Testing topic: {topic}")
        
        try:
            # Run workflow
            results = run_test_workflow(agents, topic)
            
            # Validate results
            validation = validate_results(results)
            all_validations[topic] = validation
            
            # Log validation results
            if validation["success"]:
                logger.info(f"Workflow test for '{topic}' PASSED")
            else:
                logger.warning(f"Workflow test for '{topic}' FAILED: {validation['issues']}")
            
            logger.info(f"Metrics: {validation['metrics']}")
            
        except Exception as e:
            logger.error(f"Error testing topic '{topic}': {e}")
            all_validations[topic] = {"success": False, "issues": [str(e)], "metrics": {}}
    
    # Save all validation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_file = f"./data/test_results/validation_results_{timestamp}.json"
    
    with open(validation_file, 'w') as f:
        json.dump(all_validations, f, indent=2)
    
    logger.info(f"All validation results saved to {validation_file}")
    
    # Overall summary
    success_count = sum(1 for v in all_validations.values() if v["success"])
    logger.info(f"Overall test results: {success_count}/{len(test_topics)} tests passed")
    
    return all_validations

if __name__ == "__main__":
    main()
