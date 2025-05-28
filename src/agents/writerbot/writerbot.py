"""
WriterBot agent for the AI Newsroom Team.

This module implements the WriterBot agent, responsible for transforming
structured data into coherent news articles with appropriate headlines,
style, and narrative structure.
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.writerbot")

class ArticleTemplate(BaseModel):
    """Model representing an article template."""
    name: str
    description: str
    structure: List[Dict[str, str]]
    style_guide: Dict[str, str] = Field(default_factory=dict)
    prompt_template: str


class WriterBotConfig(BaseModel):
    """Configuration for the WriterBot agent."""
    templates: List[ArticleTemplate] = Field(default_factory=list)
    default_template: str = "standard_news"
    llm_provider: str = "openai"  # Options: openai, anthropic, local
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1500
    api_key: Optional[str] = None
    style_preferences: Dict[str, Any] = Field(default_factory=dict)


class Article(BaseModel):
    """Model representing a generated article."""
    headline: str
    subheadline: Optional[str] = None
    byline: str = "AI Newsroom"
    content: str
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    template_used: str
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WriterBot:
    """
    Agent responsible for transforming structured data into coherent news articles.
    
    WriterBot takes structured information from GatherBot and generates well-written
    news articles with appropriate headlines, style, and narrative structure.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the WriterBot agent.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.templates = {t.name: t for t in self.config.templates}
        logger.info("WriterBot initialized")
    
    def _load_config(self, config_path: Optional[str]) -> WriterBotConfig:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            WriterBotConfig object.
        """
        # Default templates
        default_templates = [
            ArticleTemplate(
                name="standard_news",
                description="Standard news article with inverted pyramid structure",
                structure=[
                    {"section": "headline", "description": "Attention-grabbing headline that summarizes the main point"},
                    {"section": "lead", "description": "First paragraph covering the who, what, when, where, why, and how"},
                    {"section": "main_content", "description": "Details in descending order of importance"},
                    {"section": "background", "description": "Relevant context and history"},
                    {"section": "quotes", "description": "Relevant quotes from sources"},
                    {"section": "conclusion", "description": "Wrap-up and future implications"}
                ],
                style_guide={
                    "tone": "objective",
                    "voice": "active",
                    "sentence_length": "varied",
                    "paragraph_length": "short (2-3 sentences)"
                },
                prompt_template="""
                Write a news article about {topic} using the following information:
                
                {summary}
                
                Key facts:
                {key_facts}
                
                Quotes:
                {quotes}
                
                Sources:
                {sources}
                
                The article should follow an inverted pyramid structure with the most important information first.
                Use an objective tone and active voice. Vary sentence length but keep paragraphs short (2-3 sentences).
                Include a compelling headline.
                """
            ),
            ArticleTemplate(
                name="feature_article",
                description="In-depth feature article with narrative structure",
                structure=[
                    {"section": "headline", "description": "Creative, engaging headline"},
                    {"section": "subheadline", "description": "Explanatory subheadline that provides context"},
                    {"section": "hook", "description": "Engaging opening to draw readers in"},
                    {"section": "narrative", "description": "Story-driven explanation of the topic"},
                    {"section": "analysis", "description": "Deeper examination of implications"},
                    {"section": "human_element", "description": "Personal stories or examples"},
                    {"section": "expert_perspective", "description": "Quotes and insights from experts"},
                    {"section": "conclusion", "description": "Thoughtful closing with takeaways"}
                ],
                style_guide={
                    "tone": "conversational but authoritative",
                    "voice": "mix of active and passive",
                    "sentence_length": "varied, with some longer, more complex sentences",
                    "paragraph_length": "varied (3-5 sentences)"
                },
                prompt_template="""
                Write an in-depth feature article about {topic} using the following information:
                
                {summary}
                
                Key facts:
                {key_facts}
                
                Quotes:
                {quotes}
                
                Sources:
                {sources}
                
                The article should have a narrative structure that engages readers while providing thorough analysis.
                Use a conversational but authoritative tone. Vary sentence and paragraph length for rhythm.
                Include a creative headline and explanatory subheadline.
                """
            ),
            ArticleTemplate(
                name="breaking_news",
                description="Concise breaking news update",
                structure=[
                    {"section": "headline", "description": "Urgent, clear headline"},
                    {"section": "summary", "description": "One-paragraph summary of the breaking news"},
                    {"section": "key_details", "description": "Bullet points of essential information"},
                    {"section": "context", "description": "Brief background if necessary"},
                    {"section": "updates", "description": "Note about ongoing developments"}
                ],
                style_guide={
                    "tone": "urgent but factual",
                    "voice": "active",
                    "sentence_length": "short and direct",
                    "paragraph_length": "very short (1-2 sentences)"
                },
                prompt_template="""
                Write a breaking news update about {topic} using the following information:
                
                {summary}
                
                Key facts:
                {key_facts}
                
                Sources:
                {sources}
                
                The update should be concise and focus only on confirmed information.
                Use an urgent but factual tone with active voice and short, direct sentences.
                Include a clear headline that conveys the emergency or importance.
                Note that this is a developing story if appropriate.
                """
            )
        ]
        
        default_config = WriterBotConfig(templates=default_templates)
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                    # Convert templates from dict to ArticleTemplate objects
                    if "templates" in config_data:
                        templates = []
                        for template_data in config_data["templates"]:
                            templates.append(ArticleTemplate(**template_data))
                        config_data["templates"] = templates
                    
                    return WriterBotConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
                return default_config
        
        logger.info("Using default configuration")
        return default_config
    
    def write_article(self, gather_data: Dict[str, Any], template_name: Optional[str] = None) -> Article:
        """
        Generate an article from gathered data.
        
        Args:
            gather_data: Structured data from GatherBot.
            template_name: Optional name of template to use.
            
        Returns:
            Generated Article object.
        """
        # Use specified template or default
        template_name = template_name or self.config.default_template
        template = self.templates.get(template_name)
        
        if not template:
            logger.warning(f"Template {template_name} not found, using default")
            template = self.templates[self.config.default_template]
        
        logger.info(f"Writing article using template: {template.name}")
        
        # Extract key information from gather_data
        topic = gather_data.get("topic", "")
        summary = gather_data.get("summary", "")
        
        # Extract items
        items = gather_data.get("items", [])
        
        # Prepare key facts
        key_facts = []
        for item in items:
            if isinstance(item, dict) and "title" in item:
                key_facts.append(item["title"])
        
        # Prepare quotes (simplified)
        quotes = []
        for item in items:
            if isinstance(item, dict) and "content" in item:
                content = item["content"]
                if content and '"' in content:
                    # Very simplified quote extraction - real implementation would be more sophisticated
                    quote_start = content.find('"')
                    quote_end = content.find('"', quote_start + 1)
                    if quote_end > quote_start:
                        quote = content[quote_start:quote_end + 1]
                        source = item.get("source_name", "Unknown Source")
                        quotes.append(f"{quote} - {source}")
        
        # Prepare sources
        sources = gather_data.get("sources_used", [])
        
        # Prepare prompt variables
        prompt_vars = {
            "topic": topic,
            "summary": summary,
            "key_facts": "\n".join([f"- {fact}" for fact in key_facts]),
            "quotes": "\n".join([f"- {quote}" for quote in quotes]),
            "sources": "\n".join([f"- {source}" for source in sources])
        }
        
        # Format the prompt
        prompt = template.prompt_template.format(**prompt_vars)
        
        # Generate content using LLM
        generated_content = self._generate_with_llm(prompt)
        
        # Extract headline and content
        headline, content = self._extract_headline_and_content(generated_content)
        
        # Create article object
        article = Article(
            headline=headline,
            content=content,
            template_used=template.name,
            sources=sources,
            metadata={
                "topic": topic,
                "prompt": prompt,
                "llm_model": self.config.llm_model,
                "temperature": self.config.temperature
            }
        )
        
        logger.info(f"Article generated: {headline}")
        return article
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate content using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            Generated content.
        """
        logger.info(f"Generating content with {self.config.llm_provider} model: {self.config.llm_model}")
        
        # OpenAI implementation
        if self.config.llm_provider == "openai":
            return self._generate_with_openai(prompt)
        
        # Anthropic implementation
        elif self.config.llm_provider == "anthropic":
            return self._generate_with_anthropic(prompt)
        
        # Local model implementation (placeholder)
        elif self.config.llm_provider == "local":
            return self._generate_with_local_model(prompt)
        
        # Fallback to mock implementation for testing
        else:
            logger.warning(f"Unknown LLM provider: {self.config.llm_provider}, using mock implementation")
            return self._mock_llm_generation(prompt)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """
        Generate content using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API.
            
        Returns:
            Generated content.
        """
        try:
            # Check if API key is available
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not found")
                return self._mock_llm_generation(prompt)
            
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a professional journalist writing news articles."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # Make request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return self._mock_llm_generation(prompt)
                
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            return self._mock_llm_generation(prompt)
    
    def _generate_with_anthropic(self, prompt: str) -> str:
        """
        Generate content using Anthropic API.
        
        Args:
            prompt: The prompt to send to the API.
            
        Returns:
            Generated content.
        """
        try:
            # Check if API key is available
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Anthropic API key not found")
                return self._mock_llm_generation(prompt)
            
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.config.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            # Make request
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return self._mock_llm_generation(prompt)
                
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            return self._mock_llm_generation(prompt)
    
    def _generate_with_local_model(self, prompt: str) -> str:
        """
        Generate content using a local LLM.
        
        This is a placeholder for integration with locally hosted models.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            Generated content.
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
        
        # Extract topic from prompt
        topic = "Unknown Topic"
        if "about {topic}" in prompt:
            start_idx = prompt.find("about ") + 6
            end_idx = prompt.find(" using", start_idx)
            if end_idx > start_idx:
                topic = prompt[start_idx:end_idx]
        
        # Generate a simple mock article
        headline = f"Breaking: New Developments in {topic}"
        
        content = f"""
        {headline}
        
        In a significant development today, new information has emerged regarding {topic}. Experts are calling this a pivotal moment that could reshape our understanding of the subject.
        
        According to multiple sources, the latest findings suggest that previous assumptions about {topic} may need to be reconsidered. "This is a game-changer," said one analyst who requested anonymity due to the sensitive nature of the information.
        
        The implications of these developments are far-reaching. Industry observers note that this could affect everything from policy decisions to market dynamics in the coming months.
        
        Background:
        {topic} has been a subject of interest for many years, with researchers and analysts closely monitoring developments in this area. The recent findings build upon a body of work that dates back several decades.
        
        What's Next:
        Experts predict that we will see additional information emerge in the coming weeks as more sources come forward. "This is just the tip of the iceberg," noted one industry insider.
        
        This is a developing story and will be updated as more information becomes available.
        """
        
        return content
    
    def _extract_headline_and_content(self, generated_text: str) -> tuple[str, str]:
        """
        Extract headline and content from generated text.
        
        Args:
            generated_text: Text generated by LLM.
            
        Returns:
            Tuple of (headline, content).
        """
        lines = generated_text.strip().split('\n')
        
        # Find the headline (usually the first non-empty line)
        headline = "Untitled Article"
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.strip():
                headline = line.strip()
                content_start = i + 1
                break
        
        # Join the rest as content
        content = '\n'.join(lines[content_start:]).strip()
        
        return headline, content
    
    def generate_headline_variations(self, article: Article, count: int = 3) -> List[str]:
        """
        Generate alternative headline variations.
        
        Args:
            article: The article to generate headlines for.
            count: Number of variations to generate.
            
        Returns:
            List of headline variations.
        """
        prompt = f"""
        Generate {count} alternative headline variations for the following article:
        
        Original Headline: {article.headline}
        
        Article Content:
        {article.content[:500]}...
        
        The headlines should be attention-grabbing but accurate, and each should have a slightly different angle or tone.
        """
        
        generated_text = self._generate_with_llm(prompt)
        
        # Extract headlines from generated text
        headlines = []
        for line in generated_text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('Generate') and not line.startswith('Original'):
                # Remove numbering if present
                if '. ' in line[:4]:
                    line = line[line.find('. ') + 2:]
                headlines.append(line)
        
        # Limit to requested count
        return headlines[:count]
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the orchestrator.
        
        Args:
            task_data: Task data from orchestrator.
            
        Returns:
            Result data to be passed to the next agent.
        """
        # Extract gather data
        gather_data = task_data.get("gather_data", {})
        template_name = task_data.get("template", None)
        
        # Generate article
        article = self.write_article(gather_data, template_name)
        
        # Generate headline variations
        headline_variations = self.generate_headline_variations(article)
        
        # Convert to dictionary for message passing
        return {
            "headline": article.headline,
            "headline_variations": headline_variations,
            "content": article.content,
            "byline": article.byline,
            "sources": article.sources,
            "template_used": article.template_used,
            "created_at": article.created_at.isoformat(),
            "metadata": {
                **article.metadata,
                "topic": gather_data.get("topic", ""),
                "word_count": len(article.content.split())
            }
        }
