"""
LLM client for the AI Newsroom Team.

This module provides a client for accessing various LLM providers to generate
content for the WriterBot and FactCheckBot agents.
"""
import logging
import requests
import json
import os
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.tools.llm_client")

class LLMConfig(BaseModel):
    """Configuration for the LLM client."""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500
    request_timeout: int = 60
    retry_attempts: int = 2
    retry_delay: float = 1.0
    system_message: str = "You are a professional journalist writing news articles."


class LLMClient:
    """
    Client for accessing LLM providers.
    
    This client provides a unified interface for generating text using
    various LLM providers such as OpenAI, Anthropic, and local models.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional configuration for the client.
        """
        self.config = config or LLMConfig()
        
        # Use environment variable if API key not provided in config
        if not self.config.api_key:
            if self.config.provider == "openai":
                self.config.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.config.provider == "anthropic":
                self.config.api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        logger.info(f"LLMClient initialized with provider: {self.config.provider}")
    
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM.
            system_message: Optional system message to override the default.
            
        Returns:
            Generated text.
        """
        if self.config.provider == "openai":
            return self._generate_with_openai(prompt, system_message)
        elif self.config.provider == "anthropic":
            return self._generate_with_anthropic(prompt, system_message)
        elif self.config.provider == "local":
            return self._generate_with_local_model(prompt, system_message)
        else:
            logger.warning(f"Unknown provider: {self.config.provider}, using mock implementation")
            return self._mock_llm_generation(prompt)
    
    def _generate_with_openai(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API.
            system_message: Optional system message to override the default.
            
        Returns:
            Generated text.
        """
        if not self.config.api_key:
            logger.error("OpenAI API key not configured")
            return self._mock_llm_generation(prompt)
        
        system_msg = system_message or self.config.system_message
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        # Make request with retries
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.config.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"OpenAI API error (attempt {attempt+1}): {response.status_code} - {response.text}")
                    if attempt < self.config.retry_attempts:
                        import time
                        time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error generating with OpenAI (attempt {attempt+1}): {e}")
                if attempt < self.config.retry_attempts:
                    import time
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # If all attempts failed, return mock response
        logger.warning("All OpenAI API attempts failed, using mock implementation")
        return self._mock_llm_generation(prompt)
    
    def _generate_with_anthropic(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: The prompt to send to the API.
            system_message: Optional system message to override the default.
            
        Returns:
            Generated text.
        """
        if not self.config.api_key:
            logger.error("Anthropic API key not configured")
            return self._mock_llm_generation(prompt)
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Anthropic uses a different format for system messages
        system_msg = system_message or self.config.system_message
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "system": system_msg,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        # Make request with retries
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=self.config.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["content"][0]["text"]
                else:
                    logger.error(f"Anthropic API error (attempt {attempt+1}): {response.status_code} - {response.text}")
                    if attempt < self.config.retry_attempts:
                        import time
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Error generating with Anthropic (attempt {attempt+1}): {e}")
                if attempt < self.config.retry_attempts:
                    import time
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # If all attempts failed, return mock response
        logger.warning("All Anthropic API attempts failed, using mock implementation")
        return self._mock_llm_generation(prompt)
    
    def _generate_with_local_model(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate text using a local LLM.
        
        This is a placeholder for integration with locally hosted models.
        
        Args:
            prompt: The prompt to send to the model.
            system_message: Optional system message to override the default.
            
        Returns:
            Generated text.
        """
        # This would be implemented with libraries like llama-cpp-python
        # or other local inference frameworks
        logger.warning("Local model generation not implemented, using mock implementation")
        return self._mock_llm_generation(prompt)
    
    def _mock_llm_generation(self, prompt: str) -> str:
        """
        Generate mock content for testing purposes.
        
        Args:
            prompt: The prompt that would be sent to an LLM.
            
        Returns:
            Mock generated content.
        """
        logger.info("Using mock LLM generation")
        
        # Extract a topic from the prompt
        topic = "Unknown Topic"
        if "about " in prompt:
            start_idx = prompt.find("about ") + 6
            end_idx = prompt.find(" ", start_idx + 10)  # Look for a space after some words
            if end_idx > start_idx:
                topic = prompt[start_idx:end_idx]
        
        # Generate a simple mock article
        return f"""
        Breaking: New Developments in {topic}
        
        In a significant development today, new information has emerged regarding {topic}. Experts are calling this a pivotal moment that could reshape our understanding of the subject.
        
        According to multiple sources, the latest findings suggest that previous assumptions about {topic} may need to be reconsidered. "This is a game-changer," said one analyst who requested anonymity due to the sensitive nature of the information.
        
        The implications of these developments are far-reaching. Industry observers note that this could affect everything from policy decisions to market dynamics in the coming months.
        
        Background:
        {topic} has been a subject of interest for many years, with researchers and analysts closely monitoring developments in this area. The recent findings build upon a body of work that dates back several decades.
        
        What's Next:
        Experts predict that we will see additional information emerge in the coming weeks as more sources come forward. "This is just the tip of the iceberg," noted one industry insider.
        
        This is a developing story and will be updated as more information becomes available.
        """
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify text into provided categories.
        
        Args:
            text: Text to classify.
            categories: List of categories to classify into.
            
        Returns:
            Dictionary mapping categories to confidence scores.
        """
        # Prepare prompt
        categories_str = ", ".join(categories)
        prompt = f"""
        Classify the following text into one or more of these categories: {categories_str}
        
        Text to classify:
        {text}
        
        For each category, provide a confidence score between 0.0 and 1.0, where:
        - 0.0 means the text definitely does not belong to this category
        - 1.0 means the text definitely belongs to this category
        
        Format your response as a JSON object with categories as keys and confidence scores as values.
        """
        
        # Generate classification
        response = self.generate(prompt, "You are a text classification system.")
        
        # Try to extract JSON from response
        try:
            # Find JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                classification = json.loads(json_str)
                
                # Ensure all categories are present
                for category in categories:
                    if category not in classification:
                        classification[category] = 0.0
                
                return classification
            else:
                logger.error("Could not find JSON in classification response")
                return {category: 0.0 for category in categories}
                
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {category: 0.0 for category in categories}
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text to a specified maximum length.
        
        Args:
            text: Text to summarize.
            max_length: Maximum length of summary in words.
            
        Returns:
            Summarized text.
        """
        prompt = f"""
        Summarize the following text in no more than {max_length} words:
        
        {text}
        """
        
        return self.generate(prompt, "You are a text summarization system.")
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Text to extract key points from.
            num_points: Number of key points to extract.
            
        Returns:
            List of key points.
        """
        prompt = f"""
        Extract the {num_points} most important key points from the following text:
        
        {text}
        
        Format your response as a JSON array of strings, with each string being a key point.
        """
        
        response = self.generate(prompt, "You are a key information extraction system.")
        
        # Try to extract JSON from response
        try:
            # Find JSON-like content in the response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                key_points = json.loads(json_str)
                return key_points[:num_points]
            else:
                logger.error("Could not find JSON array in key points response")
                # Fallback: try to extract points by line
                lines = response.strip().split('\n')
                points = [line.strip() for line in lines if line.strip()]
                return points[:num_points]
                
        except Exception as e:
            logger.error(f"Error parsing key points response: {e}")
            # Fallback: try to extract points by line
            lines = response.strip().split('\n')
            points = [line.strip() for line in lines if line.strip()]
            return points[:num_points]
    
    def verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Verify a claim against provided evidence.
        
        Args:
            claim: The claim to verify.
            evidence: Evidence to check the claim against.
            
        Returns:
            Dictionary with verification results.
        """
        prompt = f"""
        Verify the following claim against the provided evidence:
        
        Claim: {claim}
        
        Evidence:
        {evidence}
        
        Determine if the claim is:
        1. Verified (fully supported by the evidence)
        2. Partially Verified (some aspects supported, others not addressed)
        3. Refuted (contradicted by the evidence)
        4. Unverifiable (evidence doesn't address the claim)
        
        Format your response as a JSON object with the following fields:
        - verification_status: One of "verified", "partially_verified", "refuted", "unverifiable"
        - confidence_score: A number between 0.0 and 1.0 indicating your confidence
        - reasoning: A brief explanation of your determination
        """
        
        response = self.generate(prompt, "You are a fact-checking system.")
        
        # Try to extract JSON from response
        try:
            # Find JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return result
            else:
                logger.error("Could not find JSON in verification response")
                return {
                    "verification_status": "unverifiable",
                    "confidence_score": 0.0,
                    "reasoning": "Could not parse verification response"
                }
                
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return {
                "verification_status": "unverifiable",
                "confidence_score": 0.0,
                "reasoning": f"Error parsing verification response: {e}"
            }
