"""
experts/news.py
---------------
NewsExpert — live news retrieval with Context Builder synthesis.

Pipeline:
  User
   │
   ▼
  Semantic Router  (routes news/current events queries here)
   │
   ▼
  News Expert
   │
   ▼
  Live Search API  (Tavily AI Search / Serper API — retrieved top 10 prioritized trusted results)
   │
   ▼
  Context Builder  (LLM RAG synthesis — merges sources, removes duplicates, resolves conflicts)
   │
   ▼
  Display Results
"""

import logging
from typing import Optional, TYPE_CHECKING, List, Dict

from config.settings import EXPERT_CONFIGS, GenerationConfig
from experts.base import BaseExpert

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

# Maximum number of live news results to fetch and display
MAX_RESULTS = 10
# Maximum snippet characters shown per article in raw fallback
MAX_SNIPPET_CHARS = 500


class NewsExpert(BaseExpert):
    """
    Expert for live news retrieval.

    Retrieves top 5-10 trusted search results and uses Context Builder RAG
    synthesis (via LLM) to merge sources, remove duplicate info, and clearly
    outline conflicting statements.
    """

    @property
    def prompt_name(self) -> str:
        return "news"

    @property
    def generation_config(self) -> GenerationConfig:
        return EXPERT_CONFIGS["news"]

    def answer(self, question: str, memory: "Optional[ConversationMemory]" = None) -> str:
        """
        Fetch live news and perform Context Builder synthesis.

        Pipeline:
          1. Call SearchService to fetch top 10 prioritized news results.
          2. Synthesize using strict RAG instructions with Groq (fallback to Gemini/OpenAI).
        """
        import time  # noqa: PLC0415
        question = self._validate_question(question)

        logger.info("NewsExpert: rewriting query and fetching live news...")

        # 1. Query Rewrite Layer
        t_rewriter_start = time.perf_counter()
        rewritten_query = question
        try:
            from news.query_rewriter import NewsQueryRewriter  # noqa: PLC0415
            rewriter = NewsQueryRewriter()
            rewritten_query = rewriter.rewrite(question)
        except Exception as rewriter_exc:
            logger.warning("NewsExpert: Query Rewriter failed: %s. Using original query.", rewriter_exc)
        rewriter_time = time.perf_counter() - t_rewriter_start

        # 2. Live Search Layer
        t_search_start = time.perf_counter()
        search_provider = "Unknown"
        try:
            from services.search_client import search_news  # noqa: PLC0415
            results = search_news(query=rewritten_query, max_results=MAX_RESULTS)
            
            # Trace search provider details
            from config.settings import TAVILY_API_KEY, SERPER_API_KEY  # noqa: PLC0415
            if TAVILY_API_KEY:
                search_provider = "Tavily Live Search"
            elif SERPER_API_KEY:
                search_provider = "Serper Google Search"
        except Exception as exc:
            logger.error("NewsExpert: Live Search API failed: %s", exc)
            return "I couldn't retrieve the latest verified information at the moment. Please try again shortly."
        search_time = time.perf_counter() - t_search_start

        if not results:
            logger.warning("NewsExpert: no reliable results returned for query: '%s'", rewritten_query)
            return "I am sorry, but additional verified information is unavailable at this time."

        # Build clean search context block for RAG
        context_block = "=== LIVE SEARCH RESULTS ===\n\n"
        for idx, r in enumerate(results, 1):
            context_block += (
                f"[Source {idx}]: {r.get('source', 'Unknown')}\n"
                f"Title: {r.get('title', 'No Title')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Content: {r.get('content', '')}\n"
                f"Published: {r.get('published', '')}\n\n"
            )

        # Context Builder instructions
        system_prompt = (
            "You are the News Expert Context Builder for IntelliMoE.\n"
            "Your task is to generate a professional, natural, and detailed response to the user's question, "
            "similar to ChatGPT, based ONLY on the provided live search results.\n\n"
            "CRITICAL INSTRUCTIONS & CONSTRAINTS:\n"
            "1. Use ONLY the retrieved search results. You must NEVER answer from your own internal static memory/knowledge.\n"
            "2. Never invent facts. Never guess. Never use prior knowledge. Do not extrapolate beyond what is explicitly stated in the results.\n"
            "3. Merge information from multiple retrieved sources seamlessly.\n"
            "4. Never rely on a single article if multiple trusted sources are available.\n"
            "5. Mention important highlights and key statistics when available.\n"
            "6. Include a timeline of events when relevant, and mention the impact of the event.\n"
            "7. Explicitly mention the sources used at the end of the response or in-text.\n"
            "8. If the retrieved search information is insufficient to answer the question, clearly state: "
            "'I am sorry, but additional verified information is unavailable at this time.'\n"
            "9. If different sources report conflicting information, clearly mention the disagreement instead of choosing one arbitrarily."
        )

        # 3. Perform LLM synthesis with robust fallback chain
        t_llm_start = time.perf_counter()
        cfg = self.generation_config
        response = None
        source_used = None

        # Try Groq (Primary)
        try:
            from services.groq_client import generate_response as groq_gen  # noqa: PLC0415
            logger.info("NewsExpert: sending request to Groq API (primary)...")
            response = groq_gen(
                prompt=f"User Question: {rewritten_query}\n\n{context_block}",
                system_prompt=system_prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=cfg.max_new_tokens
            )
            source_used = "Groq (llama-3.3-70b-versatile)"
        except Exception as e1:
            logger.warning("NewsExpert primary LLM (Groq) failed: %s. Trying fallback...", e1)

        # Try Gemini (Fallback 1)
        if not response:
            try:
                from services.gemini_client import generate_response as gemini_gen  # noqa: PLC0415
                from config.settings import GEMINI_MODEL_ID  # noqa: PLC0415
                logger.info("NewsExpert: sending request to Gemini API (fallback 1)...")
                response = gemini_gen(
                    prompt=f"User Question: {rewritten_query}\n\n{context_block}",
                    system_prompt=system_prompt,
                    model=GEMINI_MODEL_ID,
                    temperature=0.2
                )
                source_used = f"Gemini ({GEMINI_MODEL_ID})"
            except Exception as e2:
                logger.warning("NewsExpert fallback LLM (Gemini) failed: %s. Trying fallback 2...", e2)

        # Try OpenAI (Fallback 2)
        if not response:
            try:
                from services.openai_client import generate_response as openai_gen  # noqa: PLC0415
                from config.settings import OPENAI_MODEL_ID  # noqa: PLC0415
                logger.info("NewsExpert: sending request to OpenAI API (fallback 2)...")
                response = openai_gen(
                    prompt=f"User Question: {rewritten_query}\n\n{context_block}",
                    system_prompt=system_prompt,
                    model=OPENAI_MODEL_ID,
                    temperature=0.2
                )
                source_used = f"OpenAI ({OPENAI_MODEL_ID})"
            except Exception as e3:
                logger.error("NewsExpert: all LLM synthesis calls failed: %s", e3)

        llm_time = time.perf_counter() - t_llm_start

        # Store metadata attributes on self for XAI extraction
        self.last_original_query = question
        self.last_rewritten_query = rewritten_query
        self.last_search_provider = search_provider
        self.last_retrieved_articles = [
            {
                "title": r.get("title", "No Title"),
                "url": r.get("url", ""),
                "source": r.get("source", "Unknown"),
                "content": r.get("content", "")
            }
            for r in results
        ]
        self.last_sources_used = list(set(r.get("source", "Unknown") for r in results if r.get("source")))
        self.last_search_latency = search_time
        self.last_llm_latency = llm_time

        # If LLM synthesis succeeded
        if response:
            logger.info("NewsExpert: successfully synthesized response using %s.", source_used)
            # Update telemetry metrics
            prompt_len = len(rewritten_query) + len(context_block) + len(system_prompt)
            self.last_prompt_tokens = prompt_len // 4
            self.last_tokens_generated = len(response) // 4
            return response.strip()

        # Last resort fallback: Display raw results directly
        logger.warning("NewsExpert: displaying direct search results due to LLM synthesis failures.")
        return self._format_results_fallback(rewritten_query, results)

    # ------------------------------------------------------------------
    # Internal: Fallback representation
    # ------------------------------------------------------------------

    def _format_results_fallback(self, query: str, results: list[dict]) -> str:
        """Format search results directly as markdown."""
        lines = [
            f"## 📰 Live News: {query}",
            f"*{len(results)} results from Live Search*",
            "",
            "---",
        ]

        for i, article in enumerate(results, 1):
            title     = article.get("title", "No Title")
            url       = article.get("url", "")
            source    = article.get("source", "Unknown Source")
            content   = article.get("content", "").strip()[:MAX_SNIPPET_CHARS]
            published = article.get("published", "")

            if len(article.get("content", "")) > MAX_SNIPPET_CHARS:
                content += "..."

            lines.append(f"### {i}. [{title}]({url})")
            meta_parts = [f"🗞️ **{source}**"]
            if published:
                meta_parts.append(f"🕐 {published}")
            lines.append(" &nbsp;|&nbsp; ".join(meta_parts))
            lines.append("")
            if content:
                lines.append(f"> {content}")
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append(
            "*Powered by Live Web Search.*"
        )
        return "\n".join(lines)
