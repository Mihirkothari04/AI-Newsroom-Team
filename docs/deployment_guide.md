# AI Newsroom Team Deployment Guide

This document provides instructions for deploying the AI Newsroom Team system in various environments.

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for external services (News API, OpenAI, etc.)

### Installation Steps

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

4. Set up environment variables for API keys:
```bash
# Create a .env file
cat > .env << EOL
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
EOL
```

5. Run a test to verify installation:
```bash
python tests/workflow_test.py
```

## Docker Deployment

### Prerequisites

- Docker
- Docker Compose (optional, for multi-container setup)

### Using Docker

1. Build the Docker image:
```bash
docker build -t ai-newsroom-team .
```

2. Run the container with environment variables:
```bash
docker run -d \
  --name newsroom \
  -e NEWS_API_KEY=your_news_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  -v $(pwd)/data:/app/data \
  ai-newsroom-team
```

### Using Docker Compose

1. Create a `docker-compose.yml` file:
```yaml
version: '3'
services:
  newsroom:
    build: .
    environment:
      - NEWS_API_KEY=your_news_api_key
      - OPENAI_API_KEY=your_openai_api_key
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"  # If you add an API server
```

2. Start the services:
```bash
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Lambda

1. Package the application:
```bash
pip install -r requirements.txt -t ./package
cp -r src ./package/
cd package
zip -r ../deployment-package.zip .
```

2. Create a Lambda function in the AWS Console:
   - Runtime: Python 3.8+
   - Handler: lambda_handler.handler
   - Upload the deployment-package.zip file
   - Set environment variables for API keys
   - Configure memory and timeout settings (recommended: 1024MB, 5 minutes)

3. Create a `lambda_handler.py` file:
```python
import json
from src.orchestrator import NewsroomOrchestrator
from src.messaging import MessageBus

def handler(event, context):
    # Parse input
    topic = event.get('topic', 'technology')
    task_type = event.get('task_type', 'news_article')
    
    # Initialize the system
    message_bus = MessageBus()
    orchestrator = NewsroomOrchestrator(message_bus=message_bus)
    
    # Create and start a task
    task_id = orchestrator.create_task(
        topic=topic,
        task_type=task_type
    )
    orchestrator.start_task(task_id)
    
    # Wait for completion (with timeout for Lambda)
    import time
    start_time = time.time()
    max_time = context.get_remaining_time_in_millis() / 1000 - 10  # Leave 10s buffer
    
    while time.time() - start_time < max_time:
        status = orchestrator.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(2)
    
    # Get results
    if status["status"] == "completed":
        results = orchestrator.get_task_results(task_id)
        return {
            'statusCode': 200,
            'body': json.dumps({
                'headline': results['article']['headline'],
                'content': results['article']['content'],
                'accuracy': results['verification']['overall_accuracy_score']
            })
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Task failed or timed out: {status['status']}",
                'details': status
            })
        }
```

#### Using AWS ECS

1. Create an ECR repository:
```bash
aws ecr create-repository --repository-name ai-newsroom-team
```

2. Build and push the Docker image:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<region>.amazonaws.com
docker build -t <your-account-id>.dkr.ecr.<region>.amazonaws.com/ai-newsroom-team:latest .
docker push <your-account-id>.dkr.ecr.<region>.amazonaws.com/ai-newsroom-team:latest
```

3. Create an ECS cluster, task definition, and service using the AWS Console or CLI

### Google Cloud Deployment

#### Using Google Cloud Run

1. Build and push the Docker image:
```bash
gcloud builds submit --tag gcr.io/<project-id>/ai-newsroom-team
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy ai-newsroom-team \
  --image gcr.io/<project-id>/ai-newsroom-team \
  --platform managed \
  --set-env-vars NEWS_API_KEY=your_news_api_key,OPENAI_API_KEY=your_openai_api_key
```

## API Server Deployment

For applications that need to expose the AI Newsroom Team as an API service:

1. Create an API server file `api_server.py`:
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import os
from src.orchestrator import NewsroomOrchestrator
from src.messaging import MessageBus

app = FastAPI(title="AI Newsroom Team API")

# Initialize the system
message_bus = MessageBus()
orchestrator = NewsroomOrchestrator(message_bus=message_bus)

# Store task results
task_results = {}

class TaskRequest(BaseModel):
    topic: str
    task_type: str = "news_article"
    priority: str = "normal"
    params: dict = None

class TaskResponse(BaseModel):
    task_id: str
    status: str

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    task_id = orchestrator.create_task(
        topic=request.topic,
        task_type=request.task_type,
        priority=request.priority,
        params=request.params
    )
    
    # Start task in background
    background_tasks.add_task(process_task, task_id)
    
    return {"task_id": task_id, "status": "pending"}

def process_task(task_id: str):
    orchestrator.start_task(task_id)
    
    # Wait for completion
    import time
    while True:
        status = orchestrator.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(2)
    
    # Store results
    if status["status"] == "completed":
        task_results[task_id] = orchestrator.get_task_results(task_id)

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    status = orchestrator.get_task_status(task_id)
    return status

@app.get("/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    if task_id in task_results:
        return task_results[task_id]
    
    # Try to get results from orchestrator
    status = orchestrator.get_task_status(task_id)
    if status["status"] == "completed":
        results = orchestrator.get_task_results(task_id)
        task_results[task_id] = results
        return results
    
    return {"error": f"Results not available. Task status: {status['status']}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

2. Run the API server:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

3. Access the API documentation at `http://localhost:8000/docs`

## Performance Optimization

For production deployments, consider these optimizations:

1. Use a production WSGI server like Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_server:app
```

2. Implement caching for frequently requested topics:
```python
# Add to api_server.py
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_results(topic, task_type):
    # Generate and return results
    # This will cache results for identical parameters
```

3. Use a database for persistent storage of results:
```python
# Example with SQLAlchemy
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///newsroom.db")
Base = declarative_base()
Session = sessionmaker(bind=engine)

class TaskResult(Base):
    __tablename__ = "task_results"
    task_id = Column(String, primary_key=True)
    results = Column(JSON)

Base.metadata.create_all(engine)
```

## Monitoring and Logging

For production deployments, implement proper monitoring and logging:

1. Configure structured logging:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Set up logger
logger = logging.getLogger("newsroom")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

2. Implement health checks:
```python
@app.get("/health")
async def health_check():
    # Check if all components are working
    try:
        # Verify message bus is working
        message_bus.get_active_tasks()
        
        # Verify orchestrator is working
        orchestrator.get_workflow_status()
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Security Considerations

1. Store API keys securely:
   - Use environment variables for local development
   - Use AWS Secrets Manager, Google Secret Manager, or similar for cloud deployments
   - Never hardcode API keys in the codebase

2. Implement rate limiting for API endpoints:
```python
# Using FastAPI middleware
from fastapi import FastAPI, Request
import time

app = FastAPI()

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_ip):
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests = {ip: times for ip, times in self.requests.items() 
                         if any(t > minute_ago for t in times)}
        
        # Get client's requests in the last minute
        client_requests = self.requests.get(client_ip, [])
        client_requests = [t for t in client_requests if t > minute_ago]
        
        # Check if allowed
        allowed = len(client_requests) < self.requests_per_minute
        
        # Update requests
        if allowed:
            client_requests.append(now)
            self.requests[client_ip] = client_requests
        
        return allowed

rate_limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    
    return await call_next(request)
```

3. Implement authentication for API endpoints:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("API_KEY", "default-dev-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, api_key: str = Depends(verify_api_key)):
    # Protected endpoint
    # ...
```
