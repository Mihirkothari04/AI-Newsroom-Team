# AI Newsroom Team Usage Examples

This document provides practical examples of how to use the AI Newsroom Team system for various scenarios.

## Basic News Article Generation

```python
from src.orchestrator import NewsroomOrchestrator
from src.messaging import MessageBus

# Initialize the system
message_bus = MessageBus()
orchestrator = NewsroomOrchestrator(message_bus=message_bus)

# Create and start a task for a basic news article
task_id = orchestrator.create_task(
    topic="climate change",
    task_type="news_article",
    priority="normal"
)
orchestrator.start_task(task_id)

# Wait for completion and get results
import time
while True:
    status = orchestrator.get_task_status(task_id)
    print(f"Status: {status['status']} - {status.get('current_step', '')}")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Get and display results
results = orchestrator.get_task_results(task_id)
print(f"\nArticle: {results['article']['headline']}")
print(f"\n{results['article']['content']}")
print(f"\nAccuracy Score: {results['verification']['overall_accuracy_score']:.2f}")
```

## Breaking News Workflow

```python
# Create and start a task for breaking news (prioritizes speed)
task_id = orchestrator.create_task(
    topic="earthquake in japan",
    task_type="breaking_news",
    priority="high",
    params={
        "max_verification_time": 60,  # Limit verification time to 60 seconds
        "template": "breaking_news"   # Use breaking news template
    }
)
orchestrator.start_task(task_id)

# Results processing same as above
```

## In-Depth Feature Article

```python
# Create and start a task for an in-depth feature article
task_id = orchestrator.create_task(
    topic="advances in quantum computing",
    task_type="feature_article",
    priority="normal",
    params={
        "depth": "high",              # Request in-depth research
        "template": "feature_article", # Use feature article template
        "include_expert_quotes": True  # Try to include expert quotes
    }
)
orchestrator.start_task(task_id)

# Results processing same as above
```

## Custom Agent Configuration

```python
from src.agents.gatherbot import GatherBot
from src.agents.writerbot import WriterBot
from src.agents.factcheckbot import FactCheckBot
from src.tools import NewsAPIClient, LLMClient

# Create custom news API client with your API key
news_client = NewsAPIClient(config={
    "api_key": "your_news_api_key",
    "default_language": "en",
    "max_results_per_request": 15
})

# Create custom LLM client
llm_client = LLMClient(config={
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your_openai_api_key",
    "temperature": 0.5
})

# Initialize message bus
message_bus = MessageBus()

# Create custom GatherBot with specific sources
gather_bot = GatherBot(
    name="CustomGatherBot",
    message_bus=message_bus,
    config_path="config/custom_gatherbot_config.json",
    news_client=news_client  # Pass custom news client
)

# Create custom WriterBot with specific style
writer_bot = WriterBot(
    name="CustomWriterBot",
    message_bus=message_bus,
    config_path="config/custom_writerbot_config.json",
    llm_client=llm_client  # Pass custom LLM client
)

# Create custom FactCheckBot with specific verification settings
fact_check_bot = FactCheckBot(
    name="CustomFactCheckBot",
    message_bus=message_bus,
    config_path="config/custom_factcheckbot_config.json",
    llm_client=llm_client  # Pass custom LLM client
)

# Initialize orchestrator with custom agents
orchestrator = NewsroomOrchestrator(
    message_bus=message_bus,
    config_path="config/custom_orchestrator_config.json"
)

# Now use the orchestrator as in previous examples
```

## Batch Processing Multiple Topics

```python
def process_topic(topic):
    """Process a single topic and return results."""
    task_id = orchestrator.create_task(
        topic=topic,
        task_type="news_article"
    )
    orchestrator.start_task(task_id)
    
    # Wait for completion
    while True:
        status = orchestrator.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(5)
    
    # Return results
    return orchestrator.get_task_results(task_id)

# List of topics to process
topics = [
    "renewable energy",
    "artificial intelligence ethics",
    "space exploration"
]

# Process all topics
results = {}
for topic in topics:
    print(f"Processing topic: {topic}")
    results[topic] = process_topic(topic)

# Display summary of results
for topic, result in results.items():
    print(f"\nTopic: {topic}")
    print(f"Headline: {result['article']['headline']}")
    print(f"Accuracy: {result['verification']['overall_accuracy_score']:.2f}")
```

## Saving Articles to Files

```python
import os
import json
from datetime import datetime

def save_article_to_file(article_data, output_dir="./articles"):
    """Save an article to a file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from headline and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    headline = article_data["headline"].replace(" ", "_").lower()[:30]
    filename = f"{headline}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Write article to file
    with open(filepath, "w") as f:
        f.write(f"# {article_data['headline']}\n\n")
        f.write(f"By: {article_data['byline']}\n\n")
        f.write(article_data["content"])
        f.write("\n\nSources:\n")
        for source in article_data["sources"]:
            f.write(f"- {source}\n")
    
    # Also save metadata
    meta_filepath = os.path.join(output_dir, f"{headline}_{timestamp}_meta.json")
    with open(meta_filepath, "w") as f:
        json.dump(article_data["metadata"], f, indent=2)
    
    return filepath

# Example usage
task_id = orchestrator.create_task(topic="renewable energy")
orchestrator.start_task(task_id)

# Wait for completion and get results
while True:
    status = orchestrator.get_task_status(task_id)
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

results = orchestrator.get_task_results(task_id)
article_file = save_article_to_file(results["article"])
print(f"Article saved to: {article_file}")
```

## Customizing the Verification Process

```python
# Create a task with custom verification settings
task_id = orchestrator.create_task(
    topic="vaccine effectiveness",
    task_type="news_article",
    params={
        "verification": {
            "trusted_sources_only": True,  # Only use highly trusted sources
            "min_accuracy_score": 0.8,     # Require higher accuracy
            "fact_check_depth": "high",    # Do more thorough fact checking
            "require_multiple_sources": True  # Require multiple sources for claims
        }
    }
)
orchestrator.start_task(task_id)

# Results processing same as above
```

## Error Handling

```python
try:
    # Create and start a task
    task_id = orchestrator.create_task(topic="controversial topic")
    orchestrator.start_task(task_id)
    
    # Wait for completion with timeout
    import time
    start_time = time.time()
    timeout = 300  # 5 minutes
    
    while time.time() - start_time < timeout:
        status = orchestrator.get_task_status(task_id)
        
        if status["status"] == "completed":
            results = orchestrator.get_task_results(task_id)
            print(f"Task completed: {results['article']['headline']}")
            break
        elif status["status"] == "failed":
            error = status.get("error", "Unknown error")
            print(f"Task failed: {error}")
            
            # Retry with modified parameters
            if "source unavailable" in error.lower():
                print("Retrying with different sources...")
                task_id = orchestrator.create_task(
                    topic="controversial topic",
                    params={"alternative_sources": True}
                )
                orchestrator.start_task(task_id)
                # Reset timeout
                start_time = time.time()
            else:
                break
        
        time.sleep(5)
    else:
        print("Task timed out")
        orchestrator.cancel_task(task_id)

except Exception as e:
    print(f"Error: {e}")
```
