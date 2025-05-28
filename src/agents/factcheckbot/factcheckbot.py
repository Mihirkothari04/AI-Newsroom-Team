"""
FactCheckBot agent for the AI Newsroom Team.

This module implements the FactCheckBot agent, responsible for verifying factual claims,
assessing source credibility, checking for logical inconsistencies, and providing
feedback for article improvement.
"""
import logging
import json
import re
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newsroom.factcheckbot")

class SourceCredibility(BaseModel):
    """Model representing source credibility assessment."""
    name: str
    credibility_score: float  # 0.0 to 1.0
    bias_rating: str  # "left", "center", "right", "unbiased", etc.
    reliability_history: Optional[float] = None  # Historical accuracy
    transparency_score: Optional[float] = None  # Transparency about methods/funding
    notes: Optional[str] = None


class ClaimVerification(BaseModel):
    """Model representing the verification of a single claim."""
    claim_text: str
    verification_status: str  # "verified", "refuted", "partially_verified", "unverifiable"
    confidence_score: float  # 0.0 to 1.0
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sources_consulted: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ConsistencyCheck(BaseModel):
    """Model representing a consistency check within the article."""
    issue_type: str  # "contradiction", "logical_flaw", "timeline_inconsistency", etc.
    description: str
    location: Dict[str, Any]  # Information about where the issue occurs
    severity: str  # "high", "medium", "low"
    suggestion: Optional[str] = None


class VerificationResult(BaseModel):
    """Model representing the complete verification result for an article."""
    article_headline: str
    overall_accuracy_score: float  # 0.0 to 1.0
    verified_claims: List[ClaimVerification] = Field(default_factory=list)
    unverified_claims: List[ClaimVerification] = Field(default_factory=list)
    source_assessments: List[SourceCredibility] = Field(default_factory=list)
    consistency_issues: List[ConsistencyCheck] = Field(default_factory=list)
    requires_revision: bool = False
    revision_priority: str = "low"  # "high", "medium", "low"
    feedback_for_writer: Optional[str] = None
    verification_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FactCheckBotConfig(BaseModel):
    """Configuration for the FactCheckBot agent."""
    trusted_sources: List[Dict[str, Any]] = Field(default_factory=list)
    verification_threshold: float = 0.7  # Minimum confidence for "verified" status
    llm_provider: str = "openai"  # Options: openai, anthropic, local
    llm_model: str = "gpt-4"
    api_key: Optional[str] = None
    max_verification_attempts: int = 3
    web_search_enabled: bool = True
    user_agent: str = "FactCheckBot/1.0"
    request_delay: float = 1.0  # Delay between requests in seconds


class FactCheckBot:
    """
    Agent responsible for verifying factual claims and assessing content accuracy.
    
    FactCheckBot verifies claims against reliable sources, checks for logical inconsistencies,
    assesses source credibility, and provides feedback for article improvement.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the FactCheckBot agent.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        logger.info("FactCheckBot initialized")
    
    def _load_config(self, config_path: Optional[str]) -> FactCheckBotConfig:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            FactCheckBotConfig object.
        """
        # Default trusted sources
        default_trusted_sources = [
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
            },
            {
                "name": "BBC",
                "domain": "bbc.com",
                "credibility_score": 0.9,
                "bias_rating": "center-left"
            },
            {
                "name": "The New York Times",
                "domain": "nytimes.com",
                "credibility_score": 0.85,
                "bias_rating": "center-left"
            },
            {
                "name": "The Wall Street Journal",
                "domain": "wsj.com",
                "credibility_score": 0.85,
                "bias_rating": "center-right"
            }
        ]
        
        default_config = FactCheckBotConfig(trusted_sources=default_trusted_sources)
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return FactCheckBotConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
                return default_config
        
        logger.info("Using default configuration")
        return default_config
    
    def verify_article(self, article_data: Dict[str, Any], gather_data: Dict[str, Any]) -> VerificationResult:
        """
        Verify an article against the original gathered data.
        
        Args:
            article_data: Article data from WriterBot.
            gather_data: Original gathered data from GatherBot.
            
        Returns:
            VerificationResult containing assessment and feedback.
        """
        logger.info(f"Verifying article: {article_data.get('headline', 'Untitled')}")
        
        # Initialize verification result
        result = VerificationResult(
            article_headline=article_data.get("headline", "Untitled"),
            overall_accuracy_score=0.0
        )
        
        # Extract claims from article
        claims = self._extract_claims(article_data.get("content", ""))
        
        # Verify each claim
        for claim in claims:
            verification = self._verify_claim(claim, gather_data)
            
            if verification.verification_status in ["verified", "partially_verified"]:
                result.verified_claims.append(verification)
            else:
                result.unverified_claims.append(verification)
        
        # Assess source credibility
        sources = article_data.get("sources", [])
        for source in sources:
            assessment = self._assess_source_credibility(source)
            result.source_assessments.append(assessment)
        
        # Check for consistency issues
        consistency_issues = self._check_consistency(article_data.get("content", ""))
        result.consistency_issues = consistency_issues
        
        # Calculate overall accuracy score
        if claims:
            verified_score_sum = sum(c.confidence_score for c in result.verified_claims)
            unverified_penalty = sum((1 - c.confidence_score) for c in result.unverified_claims)
            
            # Weighted score calculation
            total_claims = len(claims)
            result.overall_accuracy_score = max(0.0, min(1.0, 
                (verified_score_sum - unverified_penalty) / total_claims
            ))
        
        # Determine if revision is needed
        result.requires_revision = (
            result.overall_accuracy_score < self.config.verification_threshold or
            any(issue.severity == "high" for issue in consistency_issues) or
            len(result.unverified_claims) > len(result.verified_claims)
        )
        
        # Set revision priority
        if result.requires_revision:
            if result.overall_accuracy_score < 0.5 or any(issue.severity == "high" for issue in consistency_issues):
                result.revision_priority = "high"
            elif result.overall_accuracy_score < 0.7:
                result.revision_priority = "medium"
            else:
                result.revision_priority = "low"
        
        # Generate feedback for writer
        result.feedback_for_writer = self._generate_feedback(result)
        
        logger.info(f"Verification complete. Accuracy score: {result.overall_accuracy_score:.2f}")
        logger.info(f"Requires revision: {result.requires_revision} (Priority: {result.revision_priority})")
        
        return result
    
    def _extract_claims(self, content: str) -> List[str]:
        """
        Extract factual claims from article content.
        
        Args:
            content: Article content.
            
        Returns:
            List of claim statements.
        """
        # This is a simplified implementation
        # A real implementation would use NLP to identify factual statements
        
        claims = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        for sentence in sentences:
            # Skip short sentences and questions
            if len(sentence) < 10 or sentence.endswith('?'):
                continue
                
            # Look for sentences that likely contain factual claims
            # This is a very simplified heuristic
            if any(indicator in sentence.lower() for indicator in [
                "according to", "reported", "said", "stated", "announced", "confirmed",
                "revealed", "found", "discovered", "showed", "demonstrated",
                "percent", "billion", "million", "thousand", 
                "increased", "decreased", "reduced", "grew",
                "will", "would", "could", "should"
            ]):
                claims.append(sentence)
        
        # Limit to a reasonable number of claims to verify
        return claims[:10]
    
    def _verify_claim(self, claim: str, gather_data: Dict[str, Any]) -> ClaimVerification:
        """
        Verify a single claim against gathered data and external sources.
        
        Args:
            claim: The claim to verify.
            gather_data: Original gathered data.
            
        Returns:
            ClaimVerification result.
        """
        logger.info(f"Verifying claim: {claim[:50]}...")
        
        # Initialize verification
        verification = ClaimVerification(
            claim_text=claim,
            verification_status="unverifiable",
            confidence_score=0.0
        )
        
        # First, check against gathered data
        internal_verification = self._verify_against_gathered_data(claim, gather_data)
        
        # If confident from internal data, we're done
        if internal_verification.confidence_score > 0.8:
            return internal_verification
        
        # Otherwise, check external sources if enabled
        if self.config.web_search_enabled:
            external_verification = self._verify_against_external_sources(claim)
            
            # Combine results, giving preference to higher confidence
            if external_verification.confidence_score > internal_verification.confidence_score:
                verification = external_verification
            else:
                verification = internal_verification
                
            # Combine evidence and sources
            verification.evidence.extend(external_verification.evidence)
            verification.sources_consulted.extend(external_verification.sources_consulted)
            
            # Remove duplicates
            verification.sources_consulted = list(set(verification.sources_consulted))
        else:
            verification = internal_verification
        
        return verification
    
    def _verify_against_gathered_data(self, claim: str, gather_data: Dict[str, Any]) -> ClaimVerification:
        """
        Verify a claim against the originally gathered data.
        
        Args:
            claim: The claim to verify.
            gather_data: Original gathered data.
            
        Returns:
            ClaimVerification result.
        """
        verification = ClaimVerification(
            claim_text=claim,
            verification_status="unverifiable",
            confidence_score=0.0
        )
        
        # Extract items from gather data
        items = gather_data.get("items", [])
        
        evidence = []
        sources = []
        
        # Check each item for supporting or contradicting evidence
        for item in items:
            if isinstance(item, dict):
                item_content = item.get("content", "")
                item_title = item.get("title", "")
                source_name = item.get("source_name", "Unknown Source")
                
                # Simple text matching - a real implementation would use semantic matching
                if item_content and (claim.lower() in item_content.lower() or 
                                    any(phrase in item_content.lower() for phrase in claim.lower().split())):
                    evidence.append({
                        "text": item_content,
                        "source": source_name,
                        "supports": True
                    })
                    sources.append(source_name)
                
                # Also check title
                if item_title and (claim.lower() in item_title.lower() or 
                                  any(phrase in item_title.lower() for phrase in claim.lower().split())):
                    evidence.append({
                        "text": item_title,
                        "source": source_name,
                        "supports": True
                    })
                    sources.append(source_name)
        
        # Determine verification status based on evidence
        if evidence:
            supporting_evidence = [e for e in evidence if e.get("supports", False)]
            contradicting_evidence = [e for e in evidence if not e.get("supports", False)]
            
            if supporting_evidence and not contradicting_evidence:
                verification.verification_status = "verified"
                verification.confidence_score = min(0.8, 0.5 + (len(supporting_evidence) * 0.1))
            elif supporting_evidence and contradicting_evidence:
                verification.verification_status = "partially_verified"
                verification.confidence_score = 0.5
            elif contradicting_evidence:
                verification.verification_status = "refuted"
                verification.confidence_score = min(0.8, 0.5 + (len(contradicting_evidence) * 0.1))
        
        verification.evidence = evidence
        verification.sources_consulted = list(set(sources))
        
        return verification
    
    def _verify_against_external_sources(self, claim: str) -> ClaimVerification:
        """
        Verify a claim against external sources via web search.
        
        Args:
            claim: The claim to verify.
            
        Returns:
            ClaimVerification result.
        """
        verification = ClaimVerification(
            claim_text=claim,
            verification_status="unverifiable",
            confidence_score=0.0
        )
        
        # This would typically use a search API or web scraping
        # For this implementation, we'll use a simplified approach
        
        # Extract key terms for search
        search_terms = self._extract_search_terms(claim)
        search_query = " ".join(search_terms)
        
        logger.info(f"Searching external sources for: {search_query}")
        
        # In a real implementation, this would use a search API
        # For now, we'll simulate search results
        search_results = self._simulate_search_results(search_query)
        
        evidence = []
        sources = []
        
        # Check each search result
        for result in search_results:
            source_name = result.get("source", "Unknown Source")
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Add to sources consulted
            sources.append(source_name)
            
            # Determine if content supports or contradicts claim
            # This is simplified - real implementation would use NLP
            supports = self._content_supports_claim(content, claim)
            
            if supports is not None:
                evidence.append({
                    "text": content,
                    "source": source_name,
                    "url": url,
                    "supports": supports
                })
        
        # Determine verification status based on evidence
        if evidence:
            supporting_evidence = [e for e in evidence if e.get("supports", False)]
            contradicting_evidence = [e for e in evidence if not e.get("supports", False)]
            
            # Assess source credibility for weighted decision
            credible_supporting = sum(1 for e in supporting_evidence 
                                     if self._is_credible_source(e.get("source", "")))
            credible_contradicting = sum(1 for e in contradicting_evidence 
                                        if self._is_credible_source(e.get("source", "")))
            
            if credible_supporting > credible_contradicting:
                verification.verification_status = "verified"
                verification.confidence_score = min(0.9, 0.6 + (credible_supporting * 0.1))
            elif credible_supporting and credible_contradicting:
                verification.verification_status = "partially_verified"
                verification.confidence_score = 0.5
            elif credible_contradicting:
                verification.verification_status = "refuted"
                verification.confidence_score = min(0.9, 0.6 + (credible_contradicting * 0.1))
            elif supporting_evidence and not contradicting_evidence:
                verification.verification_status = "verified"
                verification.confidence_score = 0.7
            elif supporting_evidence and contradicting_evidence:
                verification.verification_status = "partially_verified"
                verification.confidence_score = 0.4
            elif contradicting_evidence:
                verification.verification_status = "refuted"
                verification.confidence_score = 0.7
        
        verification.evidence = evidence
        verification.sources_consulted = list(set(sources))
        
        return verification
    
    def _extract_search_terms(self, claim: str) -> List[str]:
        """
        Extract key terms from a claim for search.
        
        Args:
            claim: The claim text.
            
        Returns:
            List of search terms.
        """
        # Remove common words and punctuation
        # This is simplified - real implementation would use NLP
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like", "through", "over", "before", "after", "since", "during", "above", "below", "from", "up", "down", "of", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could"}
        
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', claim.lower())
        
        # Split into words and filter out common words
        words = cleaned.split()
        terms = [word for word in words if word not in common_words and len(word) > 2]
        
        # Limit to most relevant terms
        return terms[:5]
    
    def _simulate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Simulate search results for a query.
        
        In a real implementation, this would use a search API or web scraping.
        
        Args:
            query: Search query.
            
        Returns:
            List of search result dictionaries.
        """
        # This is a placeholder that simulates search results
        # In a real implementation, this would call a search API
        
        # Generate some fake results based on the query
        results = []
        
        # Extract key terms from query
        terms = query.lower().split()
        
        # Simulate 3-5 results
        num_results = min(5, max(3, len(terms)))
        
        for i in range(num_results):
            # Rotate through some fake sources
            sources = ["Reuters", "Associated Press", "BBC News", "The New York Times", 
                      "The Washington Post", "The Guardian", "CNN", "NPR"]
            source = sources[i % len(sources)]
            
            # Generate a fake URL
            domain = source.lower().replace(" ", "")
            if not domain.endswith(".com"):
                domain += ".com"
            url = f"https://www.{domain}/article/{'-'.join(terms[:2])}-{i+1}"
            
            # Generate fake content that includes the query terms
            # In a real system, this would be actual content from the source
            content_parts = [
                f"According to recent reports from {source},",
                f"the latest information about {' '.join(terms[:3])}",
                f"indicates significant developments.",
                f"Experts at {source} have been monitoring the situation closely.",
                f"The data suggests that {' '.join(terms)} is indeed accurate.",
                f"However, some analysts have expressed caution about these findings."
            ]
            
            # Randomly include or exclude some terms to simulate varying relevance
            import random
            content = " ".join(random.sample(content_parts, k=min(4, len(content_parts))))
            
            results.append({
                "source": source,
                "url": url,
                "title": f"Latest on {' '.join(terms[:3]).title()}",
                "content": content
            })
            
        return results
    
    def _content_supports_claim(self, content: str, claim: str) -> Optional[bool]:
        """
        Determine if content supports or contradicts a claim.
        
        Args:
            content: The content text.
            claim: The claim text.
            
        Returns:
            True if supports, False if contradicts, None if neutral/unclear.
        """
        # This is a simplified implementation
        # A real implementation would use NLP for semantic understanding
        
        # Convert to lowercase for comparison
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Extract key phrases from claim
        claim_phrases = [p for p in claim_lower.split() if len(p) > 3]
        
        # Check for supporting language
        supporting_phrases = ["confirms", "supports", "shows", "demonstrates", "proves", 
                             "according to", "evidence suggests", "data indicates"]
        contradicting_phrases = ["contradicts", "refutes", "disproves", "challenges", 
                                "disputes", "contrary to", "disagrees with", "debunks"]
        
        # Count matches
        phrase_matches = sum(1 for phrase in claim_phrases if phrase in content_lower)
        
        # Check for supporting or contradicting language
        has_supporting = any(phrase in content_lower for phrase in supporting_phrases)
        has_contradicting = any(phrase in content_lower for phrase in contradicting_phrases)
        
        # Make determination
        if phrase_matches > len(claim_phrases) / 2:
            if has_contradicting:
                return False
            elif has_supporting or phrase_matches > 2:
                return True
        
        # If unclear, return None
        return None
    
    def _is_credible_source(self, source_name: str) -> bool:
        """
        Determine if a source is considered credible.
        
        Args:
            source_name: Name of the source.
            
        Returns:
            True if credible, False otherwise.
        """
        # Check against trusted sources list
        for trusted_source in self.config.trusted_sources:
            if trusted_source["name"].lower() in source_name.lower():
                return True
            
            # Also check domain if source name doesn't match
            domain = trusted_source.get("domain", "")
            if domain and domain.lower() in source_name.lower():
                return True
        
        return False
    
    def _assess_source_credibility(self, source_name: str) -> SourceCredibility:
        """
        Assess the credibility of a source.
        
        Args:
            source_name: Name of the source.
            
        Returns:
            SourceCredibility assessment.
        """
        # Check against trusted sources list
        for trusted_source in self.config.trusted_sources:
            if trusted_source["name"].lower() in source_name.lower():
                return SourceCredibility(
                    name=source_name,
                    credibility_score=trusted_source.get("credibility_score", 0.7),
                    bias_rating=trusted_source.get("bias_rating", "unknown")
                )
        
        # Default assessment for unknown sources
        return SourceCredibility(
            name=source_name,
            credibility_score=0.5,  # Neutral score for unknown sources
            bias_rating="unknown"
        )
    
    def _check_consistency(self, content: str) -> List[ConsistencyCheck]:
        """
        Check for logical inconsistencies within the article.
        
        Args:
            content: Article content.
            
        Returns:
            List of ConsistencyCheck issues.
        """
        # This is a simplified implementation
        # A real implementation would use NLP for semantic understanding
        
        issues = []
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        # Look for contradictions between paragraphs
        for i in range(len(paragraphs)):
            for j in range(i+1, len(paragraphs)):
                contradiction = self._find_contradiction(paragraphs[i], paragraphs[j])
                if contradiction:
                    issues.append(ConsistencyCheck(
                        issue_type="contradiction",
                        description=f"Contradictory statements: {contradiction}",
                        location={"paragraphs": [i, j]},
                        severity="high",
                        suggestion="Resolve the contradiction by clarifying or removing one of the statements."
                    ))
        
        # Check for timeline inconsistencies
        timeline_issue = self._check_timeline_consistency(content)
        if timeline_issue:
            issues.append(timeline_issue)
        
        return issues
    
    def _find_contradiction(self, text1: str, text2: str) -> Optional[str]:
        """
        Find contradictions between two text segments.
        
        Args:
            text1: First text segment.
            text2: Second text segment.
            
        Returns:
            Description of contradiction if found, None otherwise.
        """
        # This is a simplified implementation
        # A real implementation would use NLP for semantic understanding
        
        # Look for simple negation patterns
        negation_pairs = [
            ("increased", "decreased"),
            ("higher", "lower"),
            ("more", "less"),
            ("positive", "negative"),
            ("confirmed", "denied"),
            ("agreed", "disagreed"),
            ("approved", "rejected"),
            ("true", "false"),
            ("support", "oppose")
        ]
        
        for pos, neg in negation_pairs:
            if pos in text1.lower() and pos in text2.lower():
                # Same positive term in both texts - check context for contradiction
                pass
            elif neg in text1.lower() and neg in text2.lower():
                # Same negative term in both texts - check context for contradiction
                pass
            elif pos in text1.lower() and neg in text2.lower():
                # Potential contradiction
                return f"'{pos}' in one paragraph vs '{neg}' in another"
            elif neg in text1.lower() and pos in text2.lower():
                # Potential contradiction
                return f"'{neg}' in one paragraph vs '{pos}' in another"
        
        return None
    
    def _check_timeline_consistency(self, content: str) -> Optional[ConsistencyCheck]:
        """
        Check for timeline inconsistencies in the article.
        
        Args:
            content: Article content.
            
        Returns:
            ConsistencyCheck if issue found, None otherwise.
        """
        # This is a simplified implementation
        # A real implementation would use NLP for temporal understanding
        
        # Extract dates and time references
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b'
        dates = re.findall(date_pattern, content)
        
        # Look for "before" and "after" statements
        before_after_pattern = r'(?:before|after|prior to|following|subsequent to)\s+(?:the|this|that)'
        temporal_relations = re.findall(before_after_pattern, content.lower())
        
        # If we have multiple dates and temporal relations, check for potential issues
        if len(dates) >= 2 and temporal_relations:
            # This is a simplified check - a real implementation would parse and compare dates
            return ConsistencyCheck(
                issue_type="timeline_inconsistency",
                description="Potential timeline inconsistency detected with multiple date references",
                location={"full_article": True},
                severity="medium",
                suggestion="Review the chronology of events to ensure consistency."
            )
        
        return None
    
    def _generate_feedback(self, result: VerificationResult) -> str:
        """
        Generate feedback for the writer based on verification results.
        
        Args:
            result: VerificationResult object.
            
        Returns:
            Feedback text.
        """
        feedback_parts = []
        
        # Overall assessment
        if result.overall_accuracy_score >= 0.9:
            feedback_parts.append("Overall: Excellent accuracy. The article is well-supported by reliable sources.")
        elif result.overall_accuracy_score >= 0.7:
            feedback_parts.append("Overall: Good accuracy. Most claims are verified, with some minor issues to address.")
        elif result.overall_accuracy_score >= 0.5:
            feedback_parts.append("Overall: Moderate accuracy. Several claims require additional verification or clarification.")
        else:
            feedback_parts.append("Overall: Significant accuracy concerns. Major revision recommended.")
        
        # Unverified claims
        if result.unverified_claims:
            feedback_parts.append("\nUnverified Claims:")
            for i, claim in enumerate(result.unverified_claims, 1):
                feedback_parts.append(f"{i}. \"{claim.claim_text}\" - {claim.verification_status.capitalize()}")
                if claim.notes:
                    feedback_parts.append(f"   Note: {claim.notes}")
        
        # Consistency issues
        if result.consistency_issues:
            feedback_parts.append("\nConsistency Issues:")
            for i, issue in enumerate(result.consistency_issues, 1):
                feedback_parts.append(f"{i}. {issue.description} (Severity: {issue.severity})")
                if issue.suggestion:
                    feedback_parts.append(f"   Suggestion: {issue.suggestion}")
        
        # Source credibility
        if result.source_assessments:
            low_credibility_sources = [s for s in result.source_assessments if s.credibility_score < 0.7]
            if low_credibility_sources:
                feedback_parts.append("\nSource Credibility Concerns:")
                for source in low_credibility_sources:
                    feedback_parts.append(f"- {source.name} (Credibility Score: {source.credibility_score:.1f})")
        
        # Recommendations
        feedback_parts.append("\nRecommendations:")
        if result.requires_revision:
            if result.revision_priority == "high":
                feedback_parts.append("- Major revision required. Address all unverified claims and consistency issues.")
            elif result.revision_priority == "medium":
                feedback_parts.append("- Revision recommended. Focus on addressing the unverified claims.")
            else:
                feedback_parts.append("- Minor revisions suggested. Consider clarifying or removing unverified content.")
        else:
            feedback_parts.append("- No major revisions required. Article is ready for publication with minor edits.")
        
        return "\n".join(feedback_parts)
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned by the orchestrator.
        
        Args:
            task_data: Task data from orchestrator.
            
        Returns:
            Result data to be passed to the next agent or back to WriterBot.
        """
        # Extract article data and gather data
        article_data = task_data.get("article_data", {})
        gather_data = task_data.get("gather_data", {})
        
        # Verify the article
        verification_result = self.verify_article(article_data, gather_data)
        
        # Convert to dictionary for message passing
        return {
            "article_headline": verification_result.article_headline,
            "overall_accuracy_score": verification_result.overall_accuracy_score,
            "verified_claims": [claim.dict() for claim in verification_result.verified_claims],
            "unverified_claims": [claim.dict() for claim in verification_result.unverified_claims],
            "source_assessments": [source.dict() for source in verification_result.source_assessments],
            "consistency_issues": [issue.dict() for issue in verification_result.consistency_issues],
            "requires_revision": verification_result.requires_revision,
            "revision_priority": verification_result.revision_priority,
            "feedback_for_writer": verification_result.feedback_for_writer,
            "verification_timestamp": verification_result.verification_timestamp.isoformat(),
            "metadata": verification_result.metadata
        }
