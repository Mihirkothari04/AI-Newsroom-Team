"""
Web scraper for the AI Newsroom Team.

This module provides utilities for scraping web content to gather
information for the GatherBot agent.
"""
import logging
import requests
import time
import re
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.tools.web_scraper")

class ScraperConfig(BaseModel):
    """Configuration for the web scraper."""
    user_agent: str = "NewsroomBot/1.0 (+https://example.com/bot; bot@example.com)"
    request_timeout: int = 30
    respect_robots_txt: bool = True
    delay_between_requests: float = 1.0
    max_retries: int = 3
    follow_redirects: bool = True
    verify_ssl: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)


class WebScraper:
    """
    Web scraper for extracting content from news websites.
    
    This class provides methods for fetching and parsing web pages,
    extracting article content, and handling common scraping tasks.
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the web scraper.
        
        Args:
            config: Optional configuration for the scraper.
        """
        self.config = config or ScraperConfig()
        
        # Set default headers if not provided
        if not self.config.headers:
            self.config.headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        
        self.session = requests.Session()
        self.session.headers.update(self.config.headers)
        
        # Cache for robots.txt
        self.robots_cache = {}
        
        logger.info("WebScraper initialized")
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a web page.
        
        Args:
            url: URL to fetch.
            
        Returns:
            HTML content of the page or None if fetch failed.
        """
        # Check robots.txt if enabled
        if self.config.respect_robots_txt and not self._is_allowed(url):
            logger.warning(f"URL not allowed by robots.txt: {url}")
            return None
        
        # Fetch the page
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.request_timeout,
                    allow_redirects=self.config.follow_redirects,
                    verify=self.config.verify_ssl
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limited (429) on attempt {attempt+1}, waiting longer")
                    time.sleep(self.config.delay_between_requests * 5)  # Wait longer
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching {url} (attempt {attempt+1}): {e}")
            
            # Wait before retry
            time.sleep(self.config.delay_between_requests)
        
        return None
    
    def parse_article(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse an article from HTML content.
        
        Args:
            html: HTML content of the page.
            url: URL of the page.
            
        Returns:
            Dictionary containing parsed article data.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract domain for source name
        domain = urlparse(url).netloc
        source_name = domain.replace('www.', '')
        
        # Initialize result
        article = {
            "url": url,
            "source_name": source_name,
            "title": "",
            "author": None,
            "published_at": None,
            "content": "",
            "summary": None,
            "image_url": None,
            "metadata": {
                "domain": domain,
                "scrape_time": time.time()
            }
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            article["title"] = title_tag.text.strip()
        
        # Try to find more specific title
        for selector in ['h1.article-title', 'h1.entry-title', 'h1.title', 'h1']:
            title_elem = soup.select_one(selector)
            if title_elem:
                article["title"] = title_elem.text.strip()
                break
        
        # Extract author
        for selector in [
            'meta[name="author"]', 'a.author', '.byline', '.author', 
            '[rel="author"]', '.article-author', '.entry-author'
        ]:
            author_elem = soup.select_one(selector)
            if author_elem:
                if author_elem.name == 'meta':
                    article["author"] = author_elem.get('content', '').strip()
                else:
                    article["author"] = author_elem.text.strip()
                break
        
        # Extract publication date
        for selector in [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'time', '.date', '.published', '.article-date',
            '.entry-date', '[itemprop="datePublished"]'
        ]:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == 'meta':
                    article["published_at"] = date_elem.get('content', '')
                elif date_elem.get('datetime'):
                    article["published_at"] = date_elem.get('datetime', '')
                else:
                    article["published_at"] = date_elem.text.strip()
                break
        
        # Extract main image
        for selector in [
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            '.article-featured-image img',
            '.entry-featured-image img',
            'article img'
        ]:
            img_elem = soup.select_one(selector)
            if img_elem:
                if img_elem.name == 'meta':
                    article["image_url"] = img_elem.get('content', '')
                else:
                    article["image_url"] = img_elem.get('src', '')
                
                # Make relative URLs absolute
                if article["image_url"] and not article["image_url"].startswith(('http://', 'https://')):
                    article["image_url"] = urljoin(url, article["image_url"])
                break
        
        # Extract content
        content_selectors = [
            'article', '.article-content', '.entry-content', 
            '[itemprop="articleBody"]', '.story-body', '.post-content',
            '#content', '.content', 'main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove unwanted elements
                for unwanted in content_elem.select('script, style, nav, aside, .ad, .advertisement, .social-share'):
                    unwanted.decompose()
                
                article["content"] = content_elem.get_text(separator='\n', strip=True)
                break
        
        # If no content found, use body as fallback
        if not article["content"] and soup.body:
            article["content"] = soup.body.get_text(separator='\n', strip=True)
        
        # Extract summary/description
        for selector in [
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]',
            '.article-summary',
            '.entry-summary',
            '[itemprop="description"]'
        ]:
            summary_elem = soup.select_one(selector)
            if summary_elem:
                if summary_elem.name == 'meta':
                    article["summary"] = summary_elem.get('content', '')
                else:
                    article["summary"] = summary_elem.text.strip()
                break
        
        return article
    
    def extract_links(self, html: str, base_url: str, same_domain_only: bool = False) -> List[str]:
        """
        Extract links from HTML content.
        
        Args:
            html: HTML content.
            base_url: Base URL for resolving relative links.
            same_domain_only: Whether to only return links from the same domain.
            
        Returns:
            List of extracted links.
        """
        soup = BeautifulSoup(html, 'html.parser')
        base_domain = urlparse(base_url).netloc
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Skip empty links, anchors, and javascript
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
            
            # Make relative URLs absolute
            if not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            # Filter by domain if requested
            if same_domain_only:
                link_domain = urlparse(href).netloc
                if link_domain != base_domain:
                    continue
            
            links.append(href)
        
        return links
    
    def search_site(self, site_url: str, query: str, max_pages: int = 5) -> List[str]:
        """
        Search a site for a specific query.
        
        Args:
            site_url: Base URL of the site.
            query: Search query.
            max_pages: Maximum number of pages to check.
            
        Returns:
            List of URLs that might contain relevant content.
        """
        # Normalize site URL
        if not site_url.endswith('/'):
            site_url += '/'
        
        # Try to find search page
        search_paths = [
            f"search?q={query}",
            f"search/?q={query}",
            f"?s={query}",
            f"search?s={query}"
        ]
        
        for path in search_paths:
            search_url = urljoin(site_url, path)
            html = self.fetch_page(search_url)
            
            if html:
                # Extract links from search results
                links = self.extract_links(html, search_url, same_domain_only=True)
                
                # Filter links that might be articles
                article_links = self._filter_article_links(links, query)
                
                if article_links:
                    return article_links[:max_pages]
        
        # If search page not found or no results, try site homepage
        html = self.fetch_page(site_url)
        if html:
            links = self.extract_links(html, site_url, same_domain_only=True)
            return self._filter_article_links(links, query)[:max_pages]
        
        return []
    
    def _filter_article_links(self, links: List[str], query: str) -> List[str]:
        """
        Filter links that are likely to be articles related to the query.
        
        Args:
            links: List of links to filter.
            query: Search query.
            
        Returns:
            Filtered list of links.
        """
        # Convert query to lowercase and split into terms
        query_terms = query.lower().split()
        
        # Score links based on URL structure and query terms
        scored_links = []
        for link in links:
            score = 0
            
            # Prefer links with article-like paths
            path = urlparse(link).path
            if re.search(r'/article/', path):
                score += 3
            elif re.search(r'/news/', path):
                score += 2
            elif re.search(r'/\d{4}/\d{2}/', path):  # Date-based URLs
                score += 2
            elif re.search(r'/[^/]+/$', path) and len(path) > 10:  # Single path segment with trailing slash
                score += 1
            
            # Check for query terms in URL
            link_lower = link.lower()
            for term in query_terms:
                if term in link_lower:
                    score += 1
            
            scored_links.append((link, score))
        
        # Sort by score (descending) and return links
        scored_links.sort(key=lambda x: x[1], reverse=True)
        return [link for link, score in scored_links]
    
    def _is_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.
        
        Args:
            url: URL to check.
            
        Returns:
            True if allowed, False otherwise.
        """
        # Parse URL to get base
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check cache first
        if base_url in self.robots_cache:
            return self._check_robots_rules(url, self.robots_cache[base_url])
        
        # Fetch robots.txt
        robots_url = f"{base_url}/robots.txt"
        try:
            response = self.session.get(
                robots_url,
                timeout=self.config.request_timeout,
                verify=self.config.verify_ssl
            )
            
            if response.status_code == 200:
                # Parse robots.txt
                rules = self._parse_robots_txt(response.text)
                self.robots_cache[base_url] = rules
                return self._check_robots_rules(url, rules)
            else:
                # If robots.txt not found or error, assume allowed
                logger.warning(f"Could not fetch robots.txt for {base_url}: HTTP {response.status_code}")
                self.robots_cache[base_url] = []
                return True
                
        except Exception as e:
            logger.error(f"Error fetching robots.txt for {base_url}: {e}")
            self.robots_cache[base_url] = []
            return True
    
    def _parse_robots_txt(self, robots_txt: str) -> List[Dict[str, Any]]:
        """
        Parse robots.txt content.
        
        Args:
            robots_txt: Content of robots.txt file.
            
        Returns:
            List of rules.
        """
        rules = []
        current_agents = []
        current_rules = {"allow": [], "disallow": []}
        
        for line in robots_txt.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse directive
            if ':' in line:
                directive, value = line.split(':', 1)
                directive = directive.strip().lower()
                value = value.strip()
                
                if directive == 'user-agent':
                    # If we were already processing rules, save them
                    if current_agents and (current_rules["allow"] or current_rules["disallow"]):
                        rules.append({
                            "agents": current_agents.copy(),
                            "rules": current_rules.copy()
                        })
                        current_rules = {"allow": [], "disallow": []}
                    
                    # Start new agent section
                    if value not in current_agents:
                        current_agents.append(value)
                
                elif directive == 'disallow' and value:
                    current_rules["disallow"].append(value)
                
                elif directive == 'allow' and value:
                    current_rules["allow"].append(value)
        
        # Add the last set of rules
        if current_agents and (current_rules["allow"] or current_rules["disallow"]):
            rules.append({
                "agents": current_agents,
                "rules": current_rules
            })
        
        return rules
    
    def _check_robots_rules(self, url: str, rules: List[Dict[str, Any]]) -> bool:
        """
        Check if a URL is allowed by robots.txt rules.
        
        Args:
            url: URL to check.
            rules: Parsed robots.txt rules.
            
        Returns:
            True if allowed, False otherwise.
        """
        if not rules:
            return True
        
        parsed_url = urlparse(url)
        path = parsed_url.path
        if not path:
            path = "/"
        
        # Check for matching rules
        for rule_set in rules:
            # Check if rule applies to our user agent
            applies_to_us = False
            for agent in rule_set["agents"]:
                if agent == '*' or self.config.user_agent.lower().startswith(agent.lower()):
                    applies_to_us = True
                    break
            
            if applies_to_us:
                # Check disallow rules
                for disallow in rule_set["rules"]["disallow"]:
                    if path.startswith(disallow):
                        # Check if there's a more specific allow rule
                        for allow in rule_set["rules"]["allow"]:
                            if path.startswith(allow) and len(allow) > len(disallow):
                                return True
                        return False
        
        # If no matching disallow rule, it's allowed
        return True
