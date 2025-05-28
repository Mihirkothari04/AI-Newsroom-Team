# Complete Guide to Setting Up and Running the AI Newsroom Team System

This comprehensive guide will walk you through every step required to set up and run the AI Newsroom Team system, including all dependencies, API keys, and configuration options.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Required API Keys](#required-api-keys)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## System Requirements

### Hardware Requirements

- **CPU**: Minimum 4 cores recommended (2 cores minimum)
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Storage**: At least 2GB of free disk space
- **Internet Connection**: Required for API access and web scraping

### Software Requirements

- **Operating System**: 
  - Linux (Ubuntu 20.04+ recommended)
  - macOS 10.15+
  - Windows 10+ with WSL2 (for best compatibility)

- **Python**: Version 3.8 or higher (3.9 or 3.10 recommended)
- **Git**: For version control and updates

## Required API Keys

The AI Newsroom Team system requires the following API keys to function properly:

### 1. News API Key (Required)

The GatherBot agent uses News API to collect news articles. You need to:

1. Visit [News API](https://newsapi.org/) and create an account
2. Subscribe to a plan (they offer a free tier with limited requests)
3. Generate an API key from your dashboard
4. Note: Free tier has limitations (100 requests/day, delayed news)

### 2. OpenAI API Key (Required)

The WriterBot and FactCheckBot agents use OpenAI's models for content generation and fact verification:

1. Visit [OpenAI Platform](https://platform.openai.com/) and create an account
2. Navigate to the API section and create a new API key
3. Note your billing setup (pay-as-you-go model)
4. Recommended models: GPT-4 for best results, GPT-3.5-Turbo for lower cost

### 3. Anthropic API Key (Optional)

For alternative LLM provider:

1. Visit [Anthropic](https://www.anthropic.com/) and request API access
2. Follow their instructions to generate an API key
3. Note: This is optional and can be used as an alternative to OpenAI

### 4. SERP API Key (Optional, for enhanced web search)

For enhanced web search capabilities:

1. Visit [SerpAPI](https://serpapi.com/) and create an account
2. Generate an API key from your dashboard
3. Note: This is optional but enhances web search capabilities

## Installation Guide

### Step 1: Extract the Archive

If you received the system as a compressed archive:

```bash
# Extract the archive
tar -xzvf ai_newsroom_team.tar.gz

# Navigate to the directory
cd ai_newsroom_implementation
```

### Step 2: Set Up Python Environment

```bash
# Make sure you have Python 3.8+ installed
python --version

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

Create a `.env` file in the root directory:

```bash
# Create .env file
cat > .env << EOL
NEWS_API_KEY=your_news_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
SERP_API_KEY=your_serp_api_key_here
EOL
```

Replace the placeholder values with your actual API keys.

## Configuration

### Basic Configuration

The system uses configuration files located in the `config/` directory. Create these files if they don't exist:

#### 1. Create Orchestrator Configuration

```bash
mkdir -p config
cat > config/orchestrator_config.json << EOL
{
  "workflow_steps": {
    "news_article": [
      "gather",
      "write",
      "fact_check"
    ],
    "feature_article": [
      "gather",
      "write",
      "fact_check"
    ],
    "breaking_news": [
      "gather",
      "write"
    ]
  },
  "default_task_type": "news_article",
  "agent_timeout": 300,
  "max_retries": 3
}
EOL
```

#### 2. Create GatherBot Configuration

```bash
cat > config/gatherbot_config.json << EOL
{
  "default_sources": ["news_api", "web_scraping"],
  "max_items_per_source": 10,
  "relevance_threshold": 0.6,
  "content_filters": {
    "min_length": 100,
    "exclude_domains": []
  },
  "news_api": {
    "default_language": "en",
    "default_sort": "relevancy",
    "max_age_days": 7
  }
}
EOL
```

#### 3. Create WriterBot Configuration

```bash
cat > config/writerbot_config.json << EOL
{
  "templates": [
    {
      "name": "standard_news",
      "description": "Standard news article with inverted pyramid structure",
      "structure": [
        {"section": "headline", "description": "Attention-grabbing headline"},
        {"section": "lead", "description": "First paragraph covering the 5W1H"},
        {"section": "main_content", "description": "Details in descending order of importance"},
        {"section": "background", "description": "Relevant context and history"},
        {"section": "quotes", "description": "Relevant quotes from sources"},
        {"section": "conclusion", "description": "Wrap-up and future implications"}
      ]
    }
  ],
  "default_template": "standard_news",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "temperature": 0.7
}
EOL
```

#### 4. Create FactCheckBot Configuration

```bash
cat > config/factcheckbot_config.json << EOL
{
  "trusted_sources": [
    {
      "name": "Reuters",
      "domain": "reuters.com",
      "credibility_score": 0.95
    },
    {
      "name": "Associated Press",
      "domain": "apnews.com",
      "credibility_score": 0.95
    },
    {
      "name": "BBC",
      "domain": "bbc.com",
      "credibility_score": 0.9
    }
  ],
  "verification_threshold": 0.7,
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "max_verification_attempts": 3
}
EOL
```

### Advanced Configuration

For advanced users, you can modify these configuration files to:

- Add more trusted sources
- Change LLM models and parameters
- Adjust workflow steps
- Modify content filters
- Change verification thresholds

## Running the System

### Using the Demo Script

The simplest way to run the system is using the included demo script:

```bash
# Make sure the script is executable
chmod +x demo.sh

# Run the demo
./demo.sh
```

This will:
1. Set up the environment
2. Create necessary directories
3. Run a demonstration that creates a news article about artificial intelligence
4. Save the results to the `data/articles` directory

### Running with Custom Topics

To run the system with your own topics, you can create a simple Python script:

```bash
cat > run_custom.py << EOL
import os
import json
import time
from datetime import datetime
from src.orchestrator import NewsroomOrchestrator
from src.messaging import MessageBus

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("newsroom.custom")

def main():
    # Initialize the system
    message_bus = MessageBus()
    orchestrator = NewsroomOrchestrator(message_bus=message_bus)
    
    # Create a new task with your custom topic
    topic = "YOUR_TOPIC_HERE"  # Replace with your desired topic
    task_id = orchestrator.create_task(
        topic=topic,
        task_type="news_article",  # Options: "news_article", "feature_article", "breaking_news"
        priority="normal"  # Options: "high", "normal", "low"
    )
    
    # Start the workflow
    orchestrator.start_task(task_id)
    
    # Wait for the task to complete
    while True:
        status = orchestrator.get_task_status(task_id)
        print(f"Status: {status['status']} - Current step: {status.get('current_step', 'Unknown')}")
        
        if status["status"] in ["completed", "failed"]:
            break
        
        time.sleep(5)
    
    # Get the results
    if status["status"] == "completed":
        results = orchestrator.get_task_results(task_id)
        
        # Display article
        article = results["article"]
        print("\n" + "="*50)
        print(f"HEADLINE: {article['headline']}")
        print(f"BYLINE: {article['byline']}")
        print("="*50)
        print(article["content"])
        
        # Save article to file
        output_dir = "./data/articles"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{topic.replace(' ', '_')}_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"# {article['headline']}\n\n")
            f.write(f"By: {article['byline']}\n\n")
            f.write(article["content"])
            f.write("\n\nSources:\n")
            for source in article.get("sources", []):
                f.write(f"- {source}\n")
        
        print(f"\nArticle saved to: {filename}")
    else:
        print(f"\nTask failed: {status.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
EOL

# Run the custom script
python run_custom.py
```

Replace `YOUR_TOPIC_HERE` with your desired topic.

### Running in Production

For production environments, you can set up the system as an API service:

```bash
# Install additional dependencies
pip install gunicorn uvicorn fastapi

# Run the API server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.server:app --bind 0.0.0.0:8000
```

Then access the API at `http://localhost:8000/docs` for the Swagger UI documentation.

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues

**Symptoms**: Error messages about invalid API keys or authentication failures

**Solutions**:
- Double-check your API keys in the `.env` file
- Ensure you have sufficient credits/quota on your API accounts
- Check if your API keys have the correct permissions

```bash
# Verify API keys are loaded correctly
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('NEWS_API_KEY:', os.getenv('NEWS_API_KEY')); print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY'))"
```

#### 2. Dependency Issues

**Symptoms**: Import errors or module not found errors

**Solutions**:
- Ensure your virtual environment is activated
- Reinstall dependencies

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 3. Permission Issues

**Symptoms**: Permission denied errors when running scripts

**Solutions**:
- Make scripts executable

```bash
chmod +x demo.sh
chmod +x run_custom.py
```

#### 4. Memory Issues

**Symptoms**: System crashes or becomes unresponsive with large articles

**Solutions**:
- Reduce the number of items gathered
- Process topics in smaller batches

```bash
# Edit GatherBot configuration to reduce items
sed -i 's/"max_items_per_source": 10/"max_items_per_source": 5/' config/gatherbot_config.json
```

### Logging and Debugging

To enable detailed logging for debugging:

```bash
# Set environment variable for debug logging
export NEWSROOM_LOG_LEVEL=DEBUG

# Run with debug logging
python run_custom.py
```

## Advanced Usage

### Batch Processing Multiple Topics

```python
def process_topics(topics):
    """Process multiple topics in batch."""
    message_bus = MessageBus()
    orchestrator = NewsroomOrchestrator(message_bus=message_bus)
    
    results = {}
    for topic in topics:
        print(f"Processing topic: {topic}")
        
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
        
        if status["status"] == "completed":
            results[topic] = orchestrator.get_task_results(task_id)
    
    return results

# Example usage
topics = ["climate change", "artificial intelligence", "space exploration"]
batch_results = process_topics(topics)
```

### Customizing LLM Providers

To switch between different LLM providers:

```python
from src.tools import LLMClient

# Initialize with OpenAI
openai_client = LLMClient(config={
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your_openai_api_key",
    "temperature": 0.7
})

# Initialize with Anthropic
anthropic_client = LLMClient(config={
    "provider": "anthropic",
    "model": "claude-2",
    "api_key": "your_anthropic_api_key",
    "temperature": 0.7
})

# Use in WriterBot
from src.agents.writerbot import WriterBot
writer_bot = WriterBot(
    name="CustomWriterBot",
    message_bus=message_bus,
    llm_client=anthropic_client  # Use Anthropic instead of OpenAI
)
```

### Implementing Custom Agents

You can extend the system with custom agents:

```python
from src.agents.base_agent import Agent
from src.messaging import MessageType

class CustomAgent(Agent):
    """Custom agent implementation."""
    
    def __init__(self, name, message_bus, config_path=None):
        super().__init__(name, message_bus, config_path)
        self.register_handlers()
    
    def register_handlers(self):
        """Register message handlers."""
        self.register_handler(
            MessageType.TASK_ASSIGNMENT,
            self.handle_task_assignment
        )
    
    def handle_task_assignment(self, message):
        """Handle task assignment messages."""
        # Process the task
        result = self.process_task(message.payload)
        
        # Send result back
        self.send_message(
            target_agent=message.source_agent,
            message_type=MessageType.TASK_RESULT,
            task_id=message.task_id,
            payload=result
        )
    
    def process_task(self, task_data):
        """Process a task."""
        # Implement your custom logic here
        return {"result": "Custom processing complete"}
```

### Deploying with Docker

For containerized deployment:

```bash
# Create Dockerfile
cat > Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_custom.py"]
EOL

# Build Docker image
docker build -t ai-newsroom-team .

# Run Docker container
docker run -it --env-file .env ai-newsroom-team
```

### Setting Up a Scheduled News Service

To run the system on a schedule:

```bash
# Create a cron job script
cat > cron_newsroom.sh << EOL
#!/bin/bash
cd /path/to/ai_newsroom_implementation
source venv/bin/activate
python run_custom.py
EOL

chmod +x cron_newsroom.sh

# Add to crontab (runs daily at 6 AM)
(crontab -l 2>/dev/null; echo "0 6 * * * /path/to/ai_newsroom_implementation/cron_newsroom.sh") | crontab -
```

## Dependency Details

Here's a complete list of dependencies and their purposes:

| Dependency | Version | Purpose |
|------------|---------|---------|
| langchain | >=0.0.267 | Framework for LLM applications |
| crewai | >=0.1.0 | Multi-agent orchestration |
| openai | >=0.27.0 | OpenAI API client |
| anthropic | >=0.5.0 | Anthropic API client |
| requests | >=2.28.0 | HTTP requests for APIs |
| beautifulsoup4 | >=4.11.0 | Web scraping |
| python-dotenv | >=0.20.0 | Environment variable management |
| pydantic | >=1.9.0 | Data validation |
| fastapi | >=0.95.0 | API framework (optional) |
| uvicorn | >=0.20.0 | ASGI server (optional) |
| gunicorn | >=20.1.0 | WSGI server (optional) |
| pytest | >=7.0.0 | Testing framework |

## API Usage and Costs

### News API

- **Free Tier**: 100 requests/day, limited to headlines, 1-month-old news
- **Developer Tier**: $49/month, 500 requests/day, full articles, 3-month-old news
- **Business Tier**: Custom pricing, higher limits

### OpenAI API

- **GPT-3.5-Turbo**: $0.0015 per 1K input tokens, $0.002 per 1K output tokens
- **GPT-4**: $0.03 per 1K input tokens, $0.06 per 1K output tokens

Estimated costs per article:
- Using GPT-3.5-Turbo: ~$0.02-0.05 per article
- Using GPT-4: ~$0.30-0.70 per article

### Anthropic API

- **Claude**: $0.008 per 1K input tokens, $0.024 per 1K output tokens

### SERP API

- **Free Tier**: 100 searches/month
- **Basic Plan**: $50/month, 5,000 searches

## Final Notes

- **API Keys Security**: Never commit your API keys to version control
- **Cost Management**: Monitor your API usage to avoid unexpected charges
- **Performance Optimization**: For large-scale use, consider implementing caching
- **Updates**: Check the repository regularly for updates and improvements

For additional help or to report issues, please refer to the documentation or contact the development team.
