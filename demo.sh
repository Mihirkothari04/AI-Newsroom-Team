#!/bin/bash
# AI Newsroom Team Demonstration Script
# This script demonstrates the basic functionality of the AI Newsroom Team system

# Set up environment
echo "Setting up environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p data/messages
mkdir -p data/articles
mkdir -p data/test_results

# Run a simple demonstration
echo "Running AI Newsroom Team demonstration..."
echo "This will create a news article about artificial intelligence..."

# Create a simple demonstration script
cat > demo.py << 'EOL'
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
logger = logging.getLogger("newsroom.demo")

def main():
    print("\n===== AI NEWSROOM TEAM DEMONSTRATION =====\n")
    
    # Initialize the system
    print("Initializing AI Newsroom Team system...")
    message_bus = MessageBus()
    orchestrator = NewsroomOrchestrator(message_bus=message_bus)
    
    # Create a new task
    topic = "artificial intelligence"
    print(f"\nCreating task for topic: {topic}")
    task_id = orchestrator.create_task(
        topic=topic,
        task_type="news_article",
        priority="normal"
    )
    print(f"Task created with ID: {task_id}")
    
    # Start the workflow
    print("\nStarting the workflow...")
    orchestrator.start_task(task_id)
    
    # Wait for the task to complete
    print("\nWaiting for task to complete...")
    start_time = time.time()
    
    while True:
        status = orchestrator.get_task_status(task_id)
        current_step = status.get("current_step", "Unknown")
        
        print(f"Status: {status['status']} - Current step: {current_step}")
        
        if status["status"] in ["completed", "failed"]:
            break
        
        time.sleep(5)
    
    elapsed_time = time.time() - start_time
    print(f"\nTask completed in {elapsed_time:.2f} seconds")
    
    # Get the results
    if status["status"] == "completed":
        print("\nRetrieving results...")
        results = orchestrator.get_task_results(task_id)
        
        # Display article
        article = results["article"]
        print("\n" + "="*50)
        print(f"HEADLINE: {article['headline']}")
        print(f"BYLINE: {article['byline']}")
        print("="*50)
        print(article["content"][:500] + "...\n")
        
        # Display verification results
        verification = results["verification"]
        print(f"Accuracy Score: {verification['overall_accuracy_score']:.2f}")
        print(f"Verified Claims: {len(verification.get('verified_claims', []))}")
        print(f"Unverified Claims: {len(verification.get('unverified_claims', []))}")
        
        # Save results to file
        output_dir = "./data/articles"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/article_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"# {article['headline']}\n\n")
            f.write(f"By: {article['byline']}\n\n")
            f.write(article["content"])
            f.write("\n\nSources:\n")
            for source in article.get("sources", []):
                f.write(f"- {source}\n")
        
        print(f"\nArticle saved to: {filename}")
        
        # Save full results as JSON
        json_filename = f"{output_dir}/results_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Full results saved to: {json_filename}")
    else:
        print(f"\nTask failed: {status.get('error', 'Unknown error')}")
    
    print("\n===== DEMONSTRATION COMPLETE =====\n")

if __name__ == "__main__":
    main()
EOL

# Run the demonstration
python demo.py

echo "Demonstration complete!"
echo "You can find the generated article in the data/articles directory."
echo ""
echo "To run the demonstration again, simply execute this script:"
echo "  ./demo.sh"
echo ""
echo "For more information, please refer to the documentation:"
echo "  - README.md: Overview and basic usage"
echo "  - docs/api_documentation.md: Detailed API reference"
echo "  - docs/usage_examples.md: Code examples for various scenarios"
echo "  - docs/deployment_guide.md: Instructions for deployment"
