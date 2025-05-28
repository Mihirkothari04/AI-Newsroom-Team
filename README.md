# AI Newsroom Team

A multi-agent system for automated news gathering, writing, and fact-checking.

## Overview

The AI Newsroom Team is a collaborative multi-agent system that mirrors the workflow of a traditional newsroom. It consists of specialized agents that work together to gather information, write articles, verify facts, and prepare content for publication.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks working together
- **Automated News Gathering**: Collects information from news APIs and web sources
- **AI-Powered Content Generation**: Creates well-structured articles using LLMs
- **Automated Fact-Checking**: Verifies claims against reliable sources
- **Flexible Workflow**: Configurable pipeline for different content needs
- **Extensible Design**: Easy to add new agents or modify existing ones

## Agents

### GatherBot
Collects and organizes information from various sources:
- News API integration
- Web scraping capabilities
- Content filtering and relevance scoring
- Data structuring for downstream processing

### WriterBot
Transforms structured data into coherent articles:
- LLM integration for content generation
- Template-based article structuring
- Headline generation and optimization
- Style adaptation for different content types

### FactCheckBot
Verifies factual claims and assesses content accuracy:
- Claim verification against reliable sources
- Source credibility assessment
- Consistency checking within articles
- Feedback generation for content improvement

### EditorBot (Optional)
Polishes and prepares content for publication:
- Content refinement and enhancement
- SEO optimization
- Platform-specific formatting
- Metadata generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-newsroom-team.git
cd ai-newsroom-team
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
```bash
# Create a .env file with your API keys
echo "NEWS_API_KEY=your_news_api_key" > .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env
```

## Usage

### Basic Usage

```python
from src.orchestrator import NewsroomOrchestrator
from src.messaging import MessageBus

# Initialize the message bus
message_bus = MessageBus()

# Initialize the orchestrator
orchestrator = NewsroomOrchestrator(message_bus=message_bus)

# Create a new task
task_id = orchestrator.create_task(
    topic="artificial intelligence",
    task_type="news_article",
    priority="high"
)

# Start the workflow
orchestrator.start_task(task_id)

# Wait for the task to complete
# (In a real application, you would use async/await or callbacks)
import time
while True:
    status = orchestrator.get_task_status(task_id)
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Get the results
results = orchestrator.get_task_results(task_id)
print(f"Article: {results['article']['headline']}")
print(f"Accuracy Score: {results['verification']['overall_accuracy_score']}")
```

### Advanced Configuration

You can customize the behavior of each agent by providing configuration files:

```python
from src.agents.gatherbot import GatherBot
from src.agents.writerbot import WriterBot
from src.agents.factcheckbot import FactCheckBot

# Initialize agents with custom configurations
gather_bot = GatherBot(
    name="CustomGatherBot",
    message_bus=message_bus,
    config_path="config/gatherbot_config.json"
)

writer_bot = WriterBot(
    name="CustomWriterBot",
    message_bus=message_bus,
    config_path="config/writerbot_config.json"
)

fact_check_bot = FactCheckBot(
    name="CustomFactCheckBot",
    message_bus=message_bus,
    config_path="config/factcheckbot_config.json"
)
```

## Architecture

The system uses a message-passing architecture where agents communicate through a central message bus. Each agent has specific responsibilities:

1. **Orchestrator**: Manages the overall workflow and task assignments
2. **GatherBot**: Collects information from various sources
3. **WriterBot**: Generates article content from gathered information
4. **FactCheckBot**: Verifies facts and assesses article accuracy
5. **EditorBot** (Optional): Polishes and prepares content for publication

## Development

### Running Tests

```bash
# Run unit tests
python -m unittest tests/unit_tests.py

# Run integration tests
python -m unittest tests/integration_tests.py

# Run end-to-end workflow test
python tests/workflow_test.py
```

### Adding a New Agent

1. Create a new directory under `src/agents/`
2. Implement the agent class extending the base `Agent` class
3. Register the agent with the message bus
4. Update the orchestrator to include the new agent in the workflow

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by traditional newsroom workflows
- Built using LangChain and CrewAI concepts for agent orchestration
