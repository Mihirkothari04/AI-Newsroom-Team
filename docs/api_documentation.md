# AI Newsroom Team API Documentation

This document provides detailed information about the API for the AI Newsroom Team system.

## Table of Contents

1. [Orchestrator API](#orchestrator-api)
2. [Agent APIs](#agent-apis)
3. [Messaging System API](#messaging-system-api)
4. [Tool APIs](#tool-apis)
5. [Configuration Options](#configuration-options)

## Orchestrator API

The `NewsroomOrchestrator` class is the central controller for the AI Newsroom Team workflow.

### Methods

#### `__init__(message_bus, config_path=None)`

Initialize the orchestrator.

- **Parameters**:
  - `message_bus`: MessageBus instance for communication
  - `config_path`: Optional path to configuration file

#### `create_task(topic, task_type="news_article", priority="normal", params=None)`

Create a new task.

- **Parameters**:
  - `topic`: Main topic for the task
  - `task_type`: Type of task (e.g., "news_article", "feature", "breaking_news")
  - `priority`: Task priority ("high", "normal", "low")
  - `params`: Optional additional parameters
- **Returns**: Task ID (string)

#### `start_task(task_id)`

Start processing a task.

- **Parameters**:
  - `task_id`: ID of the task to start
- **Returns**: Boolean indicating success

#### `get_task_status(task_id)`

Get the current status of a task.

- **Parameters**:
  - `task_id`: ID of the task
- **Returns**: Dictionary with status information

#### `get_task_results(task_id)`

Get the results of a completed task.

- **Parameters**:
  - `task_id`: ID of the task
- **Returns**: Dictionary with task results

#### `cancel_task(task_id)`

Cancel a running task.

- **Parameters**:
  - `task_id`: ID of the task to cancel
- **Returns**: Boolean indicating success

## Agent APIs

### GatherBot

The `GatherBot` class is responsible for collecting information from various sources.

#### `__init__(name, message_bus, config_path=None)`

Initialize the GatherBot agent.

- **Parameters**:
  - `name`: Name of the agent instance
  - `message_bus`: MessageBus instance for communication
  - `config_path`: Optional path to configuration file

#### `process_task(task_data)`

Process a task assigned to this agent.

- **Parameters**:
  - `task_data`: Dictionary with task parameters
- **Returns**: Dictionary with gathered information

#### `gather_news(topic, sources=None, max_items=10)`

Gather news articles about a topic.

- **Parameters**:
  - `topic`: Topic to search for
  - `sources`: List of sources to use (e.g., "news_api", "web_scraping")
  - `max_items`: Maximum number of items to gather
- **Returns**: List of gathered items

### WriterBot

The `WriterBot` class transforms structured data into coherent articles.

#### `__init__(name, message_bus, config_path=None)`

Initialize the WriterBot agent.

- **Parameters**:
  - `name`: Name of the agent instance
  - `message_bus`: MessageBus instance for communication
  - `config_path`: Optional path to configuration file

#### `process_task(task_data)`

Process a task assigned to this agent.

- **Parameters**:
  - `task_data`: Dictionary with task parameters
- **Returns**: Dictionary with generated article

#### `write_article(gather_data, template_name=None)`

Generate an article from gathered data.

- **Parameters**:
  - `gather_data`: Structured data from GatherBot
  - `template_name`: Optional name of template to use
- **Returns**: Article object

#### `generate_headline_variations(article, count=3)`

Generate alternative headline variations.

- **Parameters**:
  - `article`: The article to generate headlines for
  - `count`: Number of variations to generate
- **Returns**: List of headline variations

### FactCheckBot

The `FactCheckBot` class verifies factual claims and assesses content accuracy.

#### `__init__(name, message_bus, config_path=None)`

Initialize the FactCheckBot agent.

- **Parameters**:
  - `name`: Name of the agent instance
  - `message_bus`: MessageBus instance for communication
  - `config_path`: Optional path to configuration file

#### `process_task(task_data)`

Process a task assigned to this agent.

- **Parameters**:
  - `task_data`: Dictionary with task parameters
- **Returns**: Dictionary with verification results

#### `verify_article(article_data, gather_data)`

Verify an article against the original gathered data.

- **Parameters**:
  - `article_data`: Article data from WriterBot
  - `gather_data`: Original gathered data from GatherBot
- **Returns**: VerificationResult object

## Messaging System API

### MessageBus

The `MessageBus` class handles message routing, delivery, and persistence.

#### `__init__(storage_dir="./data/messages")`

Initialize the MessageBus.

- **Parameters**:
  - `storage_dir`: Directory to store message data

#### `register_agent(agent_name, callback=None)`

Register an agent with the message bus.

- **Parameters**:
  - `agent_name`: Name of the agent
  - `callback`: Optional callback function to invoke when a message is received

#### `send_message(message)`

Send a message to the target agent.

- **Parameters**:
  - `message`: The message to send
- **Returns**: Boolean indicating success

#### `receive_message(agent_name, timeout=None)`

Receive a message for the specified agent.

- **Parameters**:
  - `agent_name`: Name of the agent
  - `timeout`: Optional timeout in seconds
- **Returns**: Message if available, None otherwise

#### `get_task_messages(task_id)`

Get all messages for a specific task.

- **Parameters**:
  - `task_id`: ID of the task
- **Returns**: List of messages for the task

#### `create_shared_memory(task_id, key, value)`

Store a value in shared memory for a task.

- **Parameters**:
  - `task_id`: ID of the task
  - `key`: Key to store the value under
  - `value`: Value to store

#### `get_shared_memory(task_id, key)`

Retrieve a value from shared memory for a task.

- **Parameters**:
  - `task_id`: ID of the task
  - `key`: Key to retrieve
- **Returns**: Stored value or None if not found

### Message

The `Message` class represents a message passed between agents.

#### `__init__(message_id=None, task_id, source_agent, target_agent, message_type, status=MessageStatus.PENDING, payload={}, instructions={}, feedback=None)`

Initialize a message.

- **Parameters**:
  - `message_id`: Optional unique identifier (auto-generated if not provided)
  - `task_id`: ID of the associated task
  - `source_agent`: Name of the sending agent
  - `target_agent`: Name of the receiving agent
  - `message_type`: Type of message (MessageType enum)
  - `status`: Status of the message (MessageStatus enum)
  - `payload`: Data payload
  - `instructions`: Optional instructions for the target agent
  - `feedback`: Optional feedback data

## Tool APIs

### NewsAPIClient

The `NewsAPIClient` class provides access to news API services.

#### `__init__(config=None)`

Initialize the NewsAPI client.

- **Parameters**:
  - `config`: Optional configuration for the client

#### `search_news(query, from_date=None, to_date=None, language=None, sort_by="relevancy", page=1)`

Search for news articles matching a query.

- **Parameters**:
  - `query`: Search query
  - `from_date`: Optional start date (YYYY-MM-DD)
  - `to_date`: Optional end date (YYYY-MM-DD)
  - `language`: Optional language code (e.g., 'en')
  - `sort_by`: Sort order ('relevancy', 'popularity', 'publishedAt')
  - `page`: Page number for pagination
- **Returns**: Dictionary containing search results

### WebScraper

The `WebScraper` class provides utilities for scraping web content.

#### `__init__(config=None)`

Initialize the web scraper.

- **Parameters**:
  - `config`: Optional configuration for the scraper

#### `fetch_page(url)`

Fetch a web page.

- **Parameters**:
  - `url`: URL to fetch
- **Returns**: HTML content of the page or None if fetch failed

#### `parse_article(html, url)`

Parse an article from HTML content.

- **Parameters**:
  - `html`: HTML content of the page
  - `url`: URL of the page
- **Returns**: Dictionary containing parsed article data

### LLMClient

The `LLMClient` class provides a unified interface for generating text using various LLM providers.

#### `__init__(config=None)`

Initialize the LLM client.

- **Parameters**:
  - `config`: Optional configuration for the client

#### `generate(prompt, system_message=None)`

Generate text using the configured LLM provider.

- **Parameters**:
  - `prompt`: The prompt to send to the LLM
  - `system_message`: Optional system message to override the default
- **Returns**: Generated text

#### `verify_claim(claim, evidence)`

Verify a claim against provided evidence.

- **Parameters**:
  - `claim`: The claim to verify
  - `evidence`: Evidence to check the claim against
- **Returns**: Dictionary with verification results

## Configuration Options

### Orchestrator Configuration

```json
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
      "fact_check",
      "edit"
    ]
  },
  "default_task_type": "news_article",
  "agent_timeout": 300,
  "max_retries": 3
}
```

### GatherBot Configuration

```json
{
  "default_sources": ["news_api", "web_scraping"],
  "max_items_per_source": 10,
  "relevance_threshold": 0.6,
  "content_filters": {
    "min_length": 100,
    "exclude_domains": ["example.com"]
  }
}
```

### WriterBot Configuration

```json
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
      ],
      "style_guide": {
        "tone": "objective",
        "voice": "active",
        "sentence_length": "varied",
        "paragraph_length": "short (2-3 sentences)"
      }
    }
  ],
  "default_template": "standard_news",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "temperature": 0.7
}
```

### FactCheckBot Configuration

```json
{
  "trusted_sources": [
    {
      "name": "Reuters",
      "domain": "reuters.com",
      "credibility_score": 0.95,
      "bias_rating": "center"
    },
    {
      "name": "Associated Press",
      "domain": "apnews.com",
      "credibility_score": 0.95,
      "bias_rating": "center"
    }
  ],
  "verification_threshold": 0.7,
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "max_verification_attempts": 3
}
```
