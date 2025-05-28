"""
News API client for the AI Newsroom Team.

This module provides a client for accessing various news APIs to gather
information for the GatherBot agent.
"""
import logging
import requests
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.tools.news_api")

class NewsAPIConfig(BaseModel):
    """Configuration for the NewsAPI client."""
    api_key: Optional[str] = None
    base_url: str = "https://newsapi.org/v2"
    default_language: str = "en"
    max_results_per_request: int = 20
    request_timeout: int = 30
    user_agent: str = "NewsroomBot/1.0"


class NewsAPIClient:
    """
    Client for accessing the NewsAPI service.
    
    This client provides methods for searching news articles, getting top headlines,
    and retrieving information about news sources.
    """
    
    def __init__(self, config: Optional[NewsAPIConfig] = None):
        """
        Initialize the NewsAPI client.
        
        Args:
            config: Optional configuration for the client.
        """
        self.config = config or NewsAPIConfig()
        
        # Use environment variable if API key not provided in config
        if not self.config.api_key:
            self.config.api_key = os.environ.get("NEWS_API_KEY")
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            "X-Api-Key": self.config.api_key or ""
        })
        
        logger.info("NewsAPIClient initialized")
    
    def search_news(self, query: str, from_date: Optional[str] = None, 
                   to_date: Optional[str] = None, language: Optional[str] = None,
                   sort_by: str = "relevancy", page: int = 1) -> Dict[str, Any]:
        """
        Search for news articles matching a query.
        
        Args:
            query: Search query.
            from_date: Optional start date (YYYY-MM-DD).
            to_date: Optional end date (YYYY-MM-DD).
            language: Optional language code (e.g., 'en').
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt').
            page: Page number for pagination.
            
        Returns:
            Dictionary containing search results.
        """
        if not self.config.api_key:
            logger.error("API key not configured")
            return {"status": "error", "message": "API key not configured"}
        
        # Set default dates if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Prepare request parameters
        params = {
            'q': query,
            'from': from_date,
            'sortBy': sort_by,
            'page': page,
            'pageSize': self.config.max_results_per_request
        }
        
        # Add optional parameters if provided
        if to_date:
            params['to'] = to_date
        if language:
            params['language'] = language
        else:
            params['language'] = self.config.default_language
        
        # Make request
        try:
            response = self.session.get(
                f"{self.config.base_url}/everything",
                params=params,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
                
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_top_headlines(self, country: Optional[str] = None, 
                         category: Optional[str] = None, 
                         query: Optional[str] = None,
                         page: int = 1) -> Dict[str, Any]:
        """
        Get top headlines.
        
        Args:
            country: Optional country code (e.g., 'us').
            category: Optional category (e.g., 'business', 'technology').
            query: Optional search query.
            page: Page number for pagination.
            
        Returns:
            Dictionary containing top headlines.
        """
        if not self.config.api_key:
            logger.error("API key not configured")
            return {"status": "error", "message": "API key not configured"}
        
        # Prepare request parameters
        params = {
            'page': page,
            'pageSize': self.config.max_results_per_request
        }
        
        # Add optional parameters if provided
        if country:
            params['country'] = country
        if category:
            params['category'] = category
        if query:
            params['q'] = query
        
        # Make request
        try:
            response = self.session.get(
                f"{self.config.base_url}/top-headlines",
                params=params,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
                
        except Exception as e:
            logger.error(f"Error getting top headlines: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_sources(self, category: Optional[str] = None, 
                   language: Optional[str] = None,
                   country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available news sources.
        
        Args:
            category: Optional category (e.g., 'business', 'technology').
            language: Optional language code (e.g., 'en').
            country: Optional country code (e.g., 'us').
            
        Returns:
            Dictionary containing news sources.
        """
        if not self.config.api_key:
            logger.error("API key not configured")
            return {"status": "error", "message": "API key not configured"}
        
        # Prepare request parameters
        params = {}
        
        # Add optional parameters if provided
        if category:
            params['category'] = category
        if language:
            params['language'] = language
        if country:
            params['country'] = country
        
        # Make request
        try:
            response = self.session.get(
                f"{self.config.base_url}/sources",
                params=params,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
                
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return {"status": "error", "message": str(e)}
    
    def format_articles(self, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format API response into a standardized article format.
        
        Args:
            api_response: Response from the NewsAPI.
            
        Returns:
            List of formatted article dictionaries.
        """
        formatted_articles = []
        
        if api_response.get("status") != "ok":
            logger.warning(f"API response status not ok: {api_response.get('status')}")
            return formatted_articles
        
        articles = api_response.get("articles", [])
        
        for article in articles:
            formatted = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source_name": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt"),
                "author": article.get("author"),
                "content": article.get("content"),
                "summary": article.get("description"),
                "image_url": article.get("urlToImage"),
                "metadata": {
                    "source_id": article.get("source", {}).get("id"),
                    "api_source": "NewsAPI"
                }
            }
            
            formatted_articles.append(formatted)
        
        return formatted_articles


class GDELTClient:
    """
    Client for accessing the GDELT Project API.
    
    This client provides methods for searching news articles and events
    from the GDELT Project's Global Knowledge Graph.
    """
    
    def __init__(self):
        """Initialize the GDELT client."""
        self.base_url = "https://api.gdeltproject.org/api/v2"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NewsroomBot/1.0"
        })
        
        logger.info("GDELTClient initialized")
    
    def search_news(self, query: str, mode: str = "artlist", 
                   format: str = "json", max_records: int = 20) -> Dict[str, Any]:
        """
        Search for news articles using the GDELT API.
        
        Args:
            query: Search query.
            mode: Search mode ('artlist', 'timelinevolume', etc.).
            format: Response format ('json', 'html', etc.).
            max_records: Maximum number of records to return.
            
        Returns:
            Dictionary containing search results.
        """
        # Prepare request parameters
        params = {
            'query': query,
            'mode': mode,
            'format': format,
            'maxrecords': max_records
        }
        
        # Make request
        try:
            response = self.session.get(
                f"{self.base_url}/doc/doc",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                if format == 'json':
                    return response.json()
                else:
                    return {"status": "ok", "content": response.text}
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
                
        except Exception as e:
            logger.error(f"Error searching GDELT news: {e}")
            return {"status": "error", "message": str(e)}
    
    def format_articles(self, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format GDELT API response into a standardized article format.
        
        Args:
            api_response: Response from the GDELT API.
            
        Returns:
            List of formatted article dictionaries.
        """
        formatted_articles = []
        
        if "articles" not in api_response:
            logger.warning("No articles found in GDELT response")
            return formatted_articles
        
        articles = api_response.get("articles", [])
        
        for article in articles:
            formatted = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source_name": article.get("domain", "Unknown"),
                "published_at": article.get("seendate"),
                "author": None,  # GDELT doesn't provide author information
                "content": None,  # GDELT doesn't provide full content
                "summary": article.get("socialimage", {}).get("description"),
                "image_url": article.get("socialimage", {}).get("url"),
                "metadata": {
                    "source_id": article.get("domain"),
                    "api_source": "GDELT",
                    "language": article.get("language"),
                    "sentiment": article.get("sentiment")
                }
            }
            
            formatted_articles.append(formatted)
        
        return formatted_articles
