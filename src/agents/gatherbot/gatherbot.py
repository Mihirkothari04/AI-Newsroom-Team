"""
GatherBot agent for the AI Newsroom Team.

This module implements the GatherBot agent, responsible for collecting and
structuring information from diverse sources such as news APIs, web pages,
and search engines.
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.gatherbot")

class NewsSource(BaseModel):
    """Model representing a news source configuration."""
    name: str
    url: str
    api_key: Optional[str] = None
    priority: int = 1
    category: Optional[str] = None
    trusted: bool = True
    scraping_enabled: bool = False
    scraping_selectors: Optional[Dict[str, str]] = None


class GatherBotConfig(BaseModel):
    """Configuration for the GatherBot agent."""
    sources: List[NewsSource] = Field(default_factory=list)
    max_articles_per_source: int = 5
    relevance_threshold: float = 0.6
    max_age_days: int = 7
    default_language: str = "en"
    user_agent: str = "NewsroomBot/1.0"
    request_delay: float = 1.0  # Delay between requests in seconds


class NewsItem(BaseModel):
    """Model representing a gathered news item."""
    title: str
    url: str
    source_name: str
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GatherResult(BaseModel):
    """Model representing the result of a gathering operation."""
    topic: str
    timestamp: datetime = Field(default_factory=datetime.now)
    items: List[NewsItem] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    key_entities: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GatherBot:
    """
    Agent responsible for collecting and structuring information from diverse sources.
    
    GatherBot monitors news feeds, scrapes web content, filters information based on
    relevance, and structures raw data into standardized formats for other agents.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GatherBot agent.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        logger.info("GatherBot initialized")
    
    def _load_config(self, config_path: Optional[str]) -> GatherBotConfig:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            GatherBotConfig object.
        """
        # Default configuration with some common news sources
        default_sources = [
            NewsSource(
                name="NewsAPI",
                url="https://newsapi.org/v2/everything",
                api_key="YOUR_API_KEY_HERE",
                priority=1,
                trusted=True
            ),
            NewsSource(
                name="New York Times",
                url="https://www.nytimes.com",
                priority=2,
                trusted=True,
                scraping_enabled=True,
                scraping_selectors={
                    "title": "h1",
                    "content": "article",
                    "author": ".byline",
                    "date": ".publish-date"
                }
            ),
            NewsSource(
                name="BBC News",
                url="https://www.bbc.com/news",
                priority=2,
                trusted=True,
                scraping_enabled=True,
                scraping_selectors={
                    "title": "h1",
                    "content": "article",
                    "author": ".byline",
                    "date": "time"
                }
            )
        ]
        
        default_config = GatherBotConfig(sources=default_sources)
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Convert dict to GatherBotConfig
                    sources = []
                    for source_data in config_data.get("sources", []):
                        sources.append(NewsSource(**source_data))
                    
                    config_data["sources"] = sources
                    return GatherBotConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
                return default_config
        
        logger.info("Using default configuration")
        return default_config
    
    def gather(self, topic: str, max_items: int = 10) -> GatherResult:
        """
        Gather news and information about a specific topic.
        
        Args:
            topic: The topic to gather information about.
            max_items: Maximum number of items to gather.
            
        Returns:
            GatherResult containing structured information.
        """
        logger.info(f"Gathering information about: {topic}")
        
        # Initialize result
        result = GatherResult(topic=topic)
        
        # Gather from news APIs
        api_items = self._gather_from_apis(topic)
        result.items.extend(api_items)
        
        # Gather from web scraping
        scraping_items = self._gather_from_scraping(topic)
        result.items.extend(scraping_items)
        
        # Filter and sort by relevance
        result.items = self._filter_and_sort_items(result.items, topic)
        
        # Limit to max_items
        result.items = result.items[:max_items]
        
        # Extract key entities and generate summary
        result.key_entities = self._extract_key_entities(result.items)
        result.summary = self._generate_summary(result.items, topic)
        
        # Record sources used
        result.sources_used = list(set(item.source_name for item in result.items))
        
        logger.info(f"Gathered {len(result.items)} items about {topic}")
        return result
    
    def _gather_from_apis(self, topic: str) -> List[NewsItem]:
        """
        Gather information from configured news APIs.
        
        Args:
            topic: The topic to search for.
            
        Returns:
            List of NewsItem objects.
        """
        items = []
        
        # Filter sources that have API keys
        api_sources = [s for s in self.config.sources if s.api_key]
        
        for source in api_sources:
            try:
                logger.info(f"Gathering from API: {source.name}")
                
                # Example for NewsAPI
                if source.name == "NewsAPI":
                    api_items = self._gather_from_newsapi(source, topic)
                    items.extend(api_items)
                
                # Add more API integrations here
                
                # Respect rate limits
                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"Error gathering from {source.name}: {e}")
        
        return items
    
    def _gather_from_newsapi(self, source: NewsSource, topic: str) -> List[NewsItem]:
        """
        Gather information from NewsAPI.
        
        Args:
            source: NewsSource configuration.
            topic: The topic to search for.
            
        Returns:
            List of NewsItem objects.
        """
        items = []
        
        # Calculate date range
        from_date = (datetime.now() - timedelta(days=self.config.max_age_days)).strftime('%Y-%m-%d')
        
        # Prepare request parameters
        params = {
            'q': topic,
            'from': from_date,
            'language': self.config.default_language,
            'sortBy': 'relevancy',
            'apiKey': source.api_key
        }
        
        # Add category if specified
        if source.category:
            params['category'] = source.category
        
        # Make request
        response = self.session.get(source.url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process articles
            for article in data.get('articles', [])[:self.config.max_articles_per_source]:
                # Convert to NewsItem
                item = NewsItem(
                    title=article.get('title', ''),
                    url=article.get('url', ''),
                    source_name=article.get('source', {}).get('name', source.name),
                    published_at=self._parse_date(article.get('publishedAt')),
                    author=article.get('author'),
                    content=article.get('content'),
                    summary=article.get('description'),
                    relevance_score=0.8,  # Placeholder - would be calculated
                    metadata={
                        'url_to_image': article.get('urlToImage'),
                        'api_source': 'NewsAPI'
                    }
                )
                
                items.append(item)
        else:
            logger.error(f"NewsAPI request failed: {response.status_code} - {response.text}")
        
        return items
    
    def _gather_from_scraping(self, topic: str) -> List[NewsItem]:
        """
        Gather information by scraping configured websites.
        
        Args:
            topic: The topic to search for.
            
        Returns:
            List of NewsItem objects.
        """
        items = []
        
        # Filter sources with scraping enabled
        scraping_sources = [s for s in self.config.sources if s.scraping_enabled]
        
        for source in scraping_sources:
            try:
                logger.info(f"Scraping from: {source.name}")
                
                # First, search for relevant pages
                search_url = f"{source.url}/search?q={topic.replace(' ', '+')}"
                response = self.session.get(search_url)
                
                if response.status_code == 200:
                    # Parse search results
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # This is a simplified example - actual selectors would depend on the site
                    article_links = soup.select('a.article-link')[:self.config.max_articles_per_source]
                    
                    # Process each article
                    for link in article_links:
                        article_url = link.get('href')
                        if not article_url.startswith('http'):
                            article_url = f"{source.url.rstrip('/')}/{article_url.lstrip('/')}"
                        
                        # Scrape the article
                        article_item = self._scrape_article(article_url, source)
                        if article_item:
                            items.append(article_item)
                        
                        # Respect rate limits
                        time.sleep(self.config.request_delay)
                
                # Respect rate limits between sources
                time.sleep(self.config.request_delay * 2)
                
            except Exception as e:
                logger.error(f"Error scraping from {source.name}: {e}")
        
        return items
    
    def _scrape_article(self, url: str, source: NewsSource) -> Optional[NewsItem]:
        """
        Scrape a single article.
        
        Args:
            url: URL of the article.
            source: NewsSource configuration.
            
        Returns:
            NewsItem if successful, None otherwise.
        """
        try:
            response = self.session.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content using configured selectors
                selectors = source.scraping_selectors or {}
                
                title = self._extract_with_selector(soup, selectors.get('title', 'h1'))
                content = self._extract_with_selector(soup, selectors.get('content', 'article'))
                author = self._extract_with_selector(soup, selectors.get('author', '.author'))
                date_str = self._extract_with_selector(soup, selectors.get('date', '.date'))
                
                # Create NewsItem
                return NewsItem(
                    title=title or "Unknown Title",
                    url=url,
                    source_name=source.name,
                    published_at=self._parse_date(date_str) if date_str else None,
                    author=author,
                    content=content,
                    relevance_score=0.7,  # Placeholder - would be calculated
                    metadata={
                        'scraped': True,
                        'scrape_time': datetime.now().isoformat()
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def _extract_with_selector(self, soup: BeautifulSoup, selector: str) -> Optional[str]:
        """
        Extract text using a CSS selector.
        
        Args:
            soup: BeautifulSoup object.
            selector: CSS selector.
            
        Returns:
            Extracted text or None.
        """
        try:
            element = soup.select_one(selector)
            return element.get_text(strip=True) if element else None
        except Exception:
            return None
    
    def _filter_and_sort_items(self, items: List[NewsItem], topic: str) -> List[NewsItem]:
        """
        Filter and sort items by relevance.
        
        Args:
            items: List of NewsItem objects.
            topic: The topic to filter by.
            
        Returns:
            Filtered and sorted list of NewsItem objects.
        """
        # Calculate relevance scores
        for item in items:
            # This is a simplified relevance calculation
            # In a real implementation, this would use more sophisticated NLP
            topic_words = set(topic.lower().split())
            title_words = set(item.title.lower().split())
            
            # Calculate overlap between topic and title
            word_overlap = len(topic_words.intersection(title_words))
            
            # Basic relevance score based on word overlap
            item.relevance_score = min(1.0, word_overlap / max(1, len(topic_words)))
            
            # Boost score for trusted sources
            source = next((s for s in self.config.sources if s.name == item.source_name), None)
            if source and source.trusted:
                item.relevance_score = min(1.0, item.relevance_score * 1.2)
        
        # Filter by relevance threshold
        filtered_items = [item for item in items if item.relevance_score >= self.config.relevance_threshold]
        
        # Sort by relevance score (descending)
        sorted_items = sorted(filtered_items, key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_items
    
    def _extract_key_entities(self, items: List[NewsItem]) -> Dict[str, List[str]]:
        """
        Extract key entities from gathered items.
        
        Args:
            items: List of NewsItem objects.
            
        Returns:
            Dictionary mapping entity types to lists of entities.
        """
        # This is a placeholder for more sophisticated entity extraction
        # In a real implementation, this would use NER models
        
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "concepts": []
        }
        
        # Simple extraction based on title and content
        for item in items:
            # Process title
            if item.title:
                words = item.title.split()
                for word in words:
                    if word[0].isupper() and len(word) > 1:
                        if word not in entities["concepts"]:
                            entities["concepts"].append(word)
            
            # Process content (simplified)
            if item.content:
                # This is very simplified - real implementation would use NLP
                pass
        
        return entities
    
    def _generate_summary(self, items: List[NewsItem], topic: str) -> str:
        """
        Generate a summary of gathered information.
        
        Args:
            items: List of NewsItem objects.
            topic: The topic of the gathering operation.
            
        Returns:
            Summary text.
        """
        # This is a placeholder for more sophisticated summarization
        # In a real implementation, this would use LLM or extractive summarization
        
        if not items:
            return f"No information found about {topic}."
        
        # Simple summary based on top items
        top_items = items[:3]
        
        summary_parts = [f"Information gathered about {topic} from {len(items)} sources:"]
        
        for i, item in enumerate(top_items, 1):
            summary_parts.append(f"{i}. {item.title} ({item.source_name})")
            if item.summary:
                summary_parts.append(f"   {item.summary}")
        
        return "\n".join(summary_parts)
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string to datetime.
        
        Args:
            date_str: Date string.
            
        Returns:
            Datetime object or None.
        """
        if not date_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try common format
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Another common format
                    return datetime.strptime(date_str, "%B %d, %Y")
                except ValueError:
                    logger.warning(f"Could not parse date: {date_str}")
                    return None
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the orchestrator.
        
        Args:
            task_data: Task data from orchestrator.
            
        Returns:
            Result data to be passed to the next agent.
        """
        topic = task_data.get("topic", "")
        max_items = task_data.get("max_items", 10)
        
        # Gather information
        gather_result = self.gather(topic, max_items)
        
        # Convert to dictionary for message passing
        return {
            "topic": gather_result.topic,
            "timestamp": gather_result.timestamp.isoformat(),
            "items": [item.dict() for item in gather_result.items],
            "sources_used": gather_result.sources_used,
            "summary": gather_result.summary,
            "key_entities": gather_result.key_entities,
            "metadata": gather_result.metadata
        }
