"""
services/search_client.py
--------------------------
Abstract Search Layer for IntelliMoE Live Search.

Implements the Dependency Inversion Principle (DIP):
  - BaseSearchProvider: Abstract interface defining the search contract.
  - TavilySearchProvider: Primary concrete search provider.
  - SerperSearchProvider: Fallback concrete search provider (using Serper Google Search/News API).
  - SearchService: Orchestrates search execution, trying Tavily first and falling back to Serper.

This ensures additional providers (e.g., Bing, DuckDuckGo) can be added in the
future by subclassing BaseSearchProvider without changing any code in NewsExpert.
"""

import logging
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from config.settings import TAVILY_API_KEY, SERPER_API_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseSearchProvider(ABC):
    """
    Abstract interface for Live Search providers.
    
    All concrete search integrations must inherit from this and implement
    the search method, returning normalized results.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 8) -> List[Dict]:
        """
        Perform web/news search and return normalized results.

        Parameters
        ----------
        query : str
            Search query topic.
        max_results : int
            Maximum number of search results to return.

        Returns
        -------
        List[Dict]
            Normalized search results where each dict contains:
              - 'title': The headline / title of the article.
              - 'url': The URL link to the original article.
              - 'source': The name/domain of the news source/publisher.
              - 'content': A snippet / description of the content.
              - 'published': Publication timestamp/date (if available).
        """
        pass


# ---------------------------------------------------------------------------
# Tavily Concrete Search Provider (Primary)
# ---------------------------------------------------------------------------

class TavilySearchProvider(BaseSearchProvider):
    """Tavily search provider optimized for AI agents and LLM context fetching."""

    def search(self, query: str, max_results: int = 8) -> List[Dict]:
        if not TAVILY_API_KEY:
            raise ValueError("Tavily API key is not configured.")

        try:
            from tavily import TavilyClient  # noqa: PLC0415
            client = TavilyClient(api_key=TAVILY_API_KEY)
            
            logger.info("TavilySearchProvider: Querying Tavily API for '%s'", query)
            response = client.search(
                query=query,
                search_depth="basic",
                topic="news",
                max_results=max_results,
                include_answer=False,
            )

            results = response.get("results", [])
            normalized = []
            for r in results:
                url = r.get("url", "")
                source = url.split("/")[2] if (url and "/" in url) else "Unknown"
                normalized.append({
                    "title": r.get("title", "No Title"),
                    "url": url,
                    "source": source,
                    "content": r.get("content", r.get("snippet", "")),
                    "published": r.get("published_date", "")
                })
            return normalized

        except ImportError:
            # Fallback to direct HTTP request to Tavily REST API to avoid hard package requirements
            logger.info("TavilySearchProvider: tavily client package not loaded. Using raw REST API request...")
            return self._search_via_rest(query, max_results)
        except Exception as e:
            logger.error("TavilySearchProvider failed: %s", e)
            raise RuntimeError(f"Tavily search failed: {e}") from e

    def _search_via_rest(self, query: str, max_results: int) -> List[Dict]:
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "topic": "news",
            "max_results": max_results
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                results = res_data.get("results", [])
                normalized = []
                for r in results:
                    r_url = r.get("url", "")
                    source = r_url.split("/")[2] if (r_url and "/" in r_url) else "Unknown"
                    normalized.append({
                        "title": r.get("title", "No Title"),
                        "url": r_url,
                        "source": source,
                        "content": r.get("content", ""),
                        "published": r.get("published_date", "")
                    })
                return normalized
        except Exception as e:
            logger.error("Tavily REST search failed: %s", e)
            raise RuntimeError(f"Tavily REST search failed: {e}") from e


# ---------------------------------------------------------------------------
# Serper Concrete Search Provider (Fallback)
# ---------------------------------------------------------------------------

class SerperSearchProvider(BaseSearchProvider):
    """Serper search provider using Google News/Search API."""

    def search(self, query: str, max_results: int = 8) -> List[Dict]:
        if not SERPER_API_KEY:
            raise ValueError("Serper API key is not configured.")

        # Try news endpoint first, fallback to organic search if no news matches
        try:
            logger.info("SerperSearchProvider: Querying Serper News API for '%s'", query)
            return self._query_serper_endpoint("https://google.serper.dev/news", "news", query, max_results)
        except Exception as e:
            logger.warning("Serper News API failed: %s. Falling back to organic search endpoint...", e)
            try:
                return self._query_serper_endpoint("https://google.serper.dev/search", "organic", query, max_results)
            except Exception as inner_exc:
                logger.error("Serper organic search endpoint also failed: %s", inner_exc)
                raise RuntimeError(f"Serper search failed: {inner_exc}") from inner_exc

    def _query_serper_endpoint(self, endpoint_url: str, key_in_response: str, query: str, max_results: int) -> List[Dict]:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "q": query,
            "num": max_results
        }
        
        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                items = res_data.get(key_in_response, [])
                
                normalized = []
                for item in items:
                    url = item.get("link", "")
                    source = item.get("source", "")
                    if not source and url:
                        source = url.split("/")[2] if "/" in url else "Unknown"
                    elif not source:
                        source = "Unknown"
                        
                    normalized.append({
                        "title": item.get("title", "No Title"),
                        "url": url,
                        "source": source,
                        "content": item.get("snippet", ""),
                        "published": item.get("date", "")
                    })
                return normalized
        except Exception as e:
            raise RuntimeError(f"Request to Serper endpoint '{endpoint_url}' failed: {e}") from e


TRUSTED_DOMAINS = [
    # Sports
    "espncricinfo.com", "cricbuzz.com", "icc-cricket.com",
    # General News
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com",
    # Technology
    "techcrunch.com", "theverge.com",
    # Finance
    "bloomberg.com", "finance.yahoo.com"
]

UNRELIABLE_DOMAINS = [
    "blogspot.com", "wordpress.com", "reddit.com", "quora.com", 
    "pinterest.com", "tumblr.com", "youtube.com", "twitter.com",
    "facebook.com", "instagram.com", "tiktok.com", "medium.com",
    "clickbait", "spam", "adware", "malware", "forum", "blog"
]


class SearchService:
    """Orchestrates multi-provider search with automatic primary -> fallback routing."""

    def __init__(self, providers: Optional[List[BaseSearchProvider]] = None) -> None:
        if providers:
            self._providers = providers
        else:
            # Default pipeline order: Tavily (primary) -> Serper (fallback)
            self._providers = [TavilySearchProvider(), SerperSearchProvider()]

    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Removes duplicate articles by URL or matching titles."""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        for r in results:
            url = r.get("url", "").strip().lower()
            title = r.get("title", "").strip().lower()
            
            # Clean title to compare alphanumeric characters only
            title_clean = "".join(c for c in title if c.isalnum())
            
            if not url or not title_clean:
                continue
            if url in seen_urls or title_clean in seen_titles:
                continue
                
            seen_urls.add(url)
            seen_titles.add(title_clean)
            unique_results.append(r)
        return unique_results

    def filter_unreliable_sources(self, results: List[Dict]) -> List[Dict]:
        """Filters out results originating from blacklisted or low-quality domains."""
        clean_results = []
        for r in results:
            url = r.get("url", "").lower()
            is_unreliable = False
            for domain in UNRELIABLE_DOMAINS:
                if domain in url:
                    is_unreliable = True
                    break
            if not is_unreliable:
                clean_results.append(r)
        return clean_results

    def prioritize_results(self, results: List[Dict]) -> List[Dict]:
        """Sorts results so that those from trusted domains are positioned first."""
        def get_priority(item: Dict) -> int:
            url = item.get("url", "").lower()
            source = item.get("source", "").lower()
            for domain in TRUSTED_DOMAINS:
                if domain in url or domain in source:
                    return 0  # Higher priority (first)
            return 1  # Normal priority
        return sorted(results, key=get_priority)

    def search_news(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Fetch news from search providers in priority order.
        
        Returns the first successful result list from the configured providers,
        filtering duplicates/spam and prioritizing trusted sources.
        """
        errors = []
        for provider in self._providers:
            provider_name = provider.__class__.__name__
            try:
                # Retrieve slightly more raw results to have a better pool for prioritization and filtering
                raw_results = provider.search(query, max_results=max(max_results * 2, 15))
                
                # 1. Remove duplicate articles
                deduped = self.deduplicate_results(raw_results)
                
                # 2. Ignore low-quality or unreliable websites
                filtered = self.filter_unreliable_sources(deduped)
                
                # 3. Prioritize official and trusted sources
                prioritized = self.prioritize_results(filtered)
                
                final_results = prioritized[:max_results]
                logger.info(
                    "SearchService: Successfully retrieved, validated, and prioritized %d results from %s.",
                    len(final_results),
                    provider_name
                )
                return final_results
            except Exception as e:
                logger.warning("SearchService: Provider %s failed: %s", provider_name, e)
                errors.append(f"{provider_name}: {e}")

        # If all providers failed, raise an aggregate exception
        error_msg = " | ".join(errors)
        raise RuntimeError(f"All search providers failed. Details: {error_msg}")


# Backward-compatible simple functional entrypoint
def search_news(query: str, max_results: int = 10) -> List[Dict]:
    """Simple helper function matching the legacy search_news signature."""
    service = SearchService()
    return service.search_news(query, max_results=max_results)


