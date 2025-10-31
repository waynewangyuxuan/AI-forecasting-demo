"""
Query Generation Service for AI Forecasting Pipeline.

Generates diverse, optimized search queries for comprehensive information
retrieval using LLM-based query expansion and diversification.
"""

import re
from typing import List, Optional, Set
import structlog
from difflib import SequenceMatcher

from config.settings import settings
from db.models import SearchQuery
from services.llm_client import LLMClient, create_llm_client

logger = structlog.get_logger(__name__)


# Prompt version for tracking
QUERY_GENERATION_PROMPT_VERSION = "v1.0"


def get_query_generation_prompt(question: str, prior_queries: Optional[List[str]] = None) -> str:
    """
    Build prompt for LLM-based query generation.

    The prompt is optimized for forecasting use cases with emphasis on:
    - Temporal specificity (dates, recent developments)
    - Multiple perspectives (technical, market, regulatory, geopolitical)
    - Diverse terminology and phrasing
    - Avoiding redundancy with prior queries

    Args:
        question: The forecasting question
        prior_queries: List of previously generated queries to avoid duplication

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a research assistant helping generate diverse search queries for forecasting.

Given a forecasting question, generate 10 optimized Google search queries that will help gather comprehensive information to answer the question. The queries should:

1. **Cover multiple angles**: technical details, market trends, regulatory developments, expert opinions, historical context
2. **Include temporal specificity**: mention relevant years, "recent", "latest", "2024", "2025" etc.
3. **Use varied terminology**: different phrasings, synonyms, related concepts
4. **Be specific and targeted**: avoid generic queries, focus on concrete aspects
5. **Be optimized for search engines**: natural language, 3-8 words typically

**Forecasting Question:**
{question}

**Requirements for your queries:**
- Generate EXACTLY 10 queries
- Each query should explore a different aspect or angle
- Queries should be diverse and minimize overlap
- Include specific time references where relevant
- Use quotes for exact phrases when needed
- Consider: who, what, when, where, why, how

"""

    if prior_queries and len(prior_queries) > 0:
        prior_list = "\n".join(f"- {q}" for q in prior_queries)
        prompt += f"""
**IMPORTANT: Avoid duplication with these prior queries:**
{prior_list}

Make sure your new queries explore different angles and use different terminology.

"""

    prompt += """
**Output format (JSON):**
Return a JSON object with a "queries" array containing exactly 10 query strings.

Example:
{
  "queries": [
    "China humanoid robot mass production 2025",
    "BYD Xiaomi robot manufacturing capacity timeline",
    "chinese robotics industry production forecasts 2024-2025",
    ...
  ]
}

Generate the queries now:"""

    return prompt


class QueryGenerator:
    """
    Service for generating diverse search queries from forecasting questions.

    Uses LLM to generate queries with multiple perspectives, then applies
    post-processing for deduplication and validation.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_queries: Optional[int] = None,
    ):
        """
        Initialize the query generator.

        Args:
            llm_client: LLM client to use (defaults to GeminiClient)
            max_queries: Maximum number of queries to generate
        """
        self.llm_client = llm_client or create_llm_client(provider=settings.llm_provider)
        self.max_queries = max_queries or settings.max_search_queries
        self.logger = structlog.get_logger(__name__)

    def generate_queries(
        self,
        question: str,
        run_id: int,
        prior_queries: Optional[List[str]] = None,
    ) -> List[SearchQuery]:
        """
        Generate diverse search queries for a forecasting question.

        Args:
            question: The forecasting question text
            run_id: Run ID to associate with generated queries
            prior_queries: Previously generated queries to avoid duplication

        Returns:
            List of SearchQuery objects ready for persistence

        Raises:
            Exception: If query generation fails
        """
        self.logger.info(
            "query_generation_start",
            question_length=len(question),
            run_id=run_id,
            prior_query_count=len(prior_queries) if prior_queries else 0,
        )

        try:
            # Generate queries using LLM
            raw_queries = self._generate_with_llm(question, prior_queries)

            # Post-process queries
            processed_queries = self._post_process_queries(
                raw_queries,
                question,
                prior_queries or [],
            )

            # Convert to SearchQuery objects
            search_queries = [
                SearchQuery(
                    run_id=run_id,
                    query_text=query,
                    prompt_version=QUERY_GENERATION_PROMPT_VERSION,
                )
                for query in processed_queries
            ]

            self.logger.info(
                "query_generation_complete",
                run_id=run_id,
                raw_count=len(raw_queries),
                processed_count=len(search_queries),
            )

            return search_queries

        except Exception as e:
            self.logger.error(
                "query_generation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _generate_with_llm(
        self,
        question: str,
        prior_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate queries using the LLM.

        Args:
            question: Forecasting question
            prior_queries: Prior queries to avoid

        Returns:
            List of raw query strings from LLM

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        # Build prompt
        prompt = get_query_generation_prompt(question, prior_queries)

        # Call LLM with JSON output
        response = self.llm_client.generate_json(
            prompt=prompt,
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=1000,
        )

        # Extract queries from response
        if "queries" not in response:
            raise ValueError("LLM response missing 'queries' field")

        queries = response["queries"]

        if not isinstance(queries, list):
            raise ValueError("LLM 'queries' field is not a list")

        if len(queries) == 0:
            raise ValueError("LLM returned empty queries list")

        self.logger.debug(
            "llm_queries_generated",
            count=len(queries),
        )

        return queries

    def _post_process_queries(
        self,
        queries: List[str],
        question: str,
        prior_queries: List[str],
    ) -> List[str]:
        """
        Post-process and validate generated queries.

        Applies:
        - Deduplication (case-insensitive, fuzzy matching)
        - Length validation (too short or too long)
        - Keyword coverage validation
        - Limit to max_queries

        Args:
            queries: Raw queries from LLM
            question: Original question for keyword extraction
            prior_queries: Prior queries to check against

        Returns:
            Processed and validated query list
        """
        processed = []
        seen = set()  # Case-folded queries for exact deduplication
        seen_prior = {q.lower().strip() for q in prior_queries}

        # Extract keywords from question for coverage check
        question_keywords = self._extract_keywords(question)

        for query in queries:
            # Clean query
            query = query.strip()

            # Skip if empty
            if not query:
                continue

            # Length validation (3-150 characters is reasonable)
            if len(query) < 3:
                self.logger.debug("query_too_short", query=query)
                continue
            if len(query) > 150:
                self.logger.debug("query_too_long", query=query)
                continue

            # Exact deduplication (case-insensitive)
            query_lower = query.lower()
            if query_lower in seen or query_lower in seen_prior:
                self.logger.debug("query_duplicate", query=query)
                continue

            # Fuzzy deduplication with existing queries
            if self._is_too_similar(query, processed + prior_queries):
                self.logger.debug("query_too_similar", query=query)
                continue

            # Keyword coverage check (at least one keyword from question)
            if not self._has_keyword_coverage(query, question_keywords):
                self.logger.debug("query_low_keyword_coverage", query=query)
                continue

            # Add to processed list
            processed.append(query)
            seen.add(query_lower)

            # Stop if we have enough queries
            if len(processed) >= self.max_queries:
                break

        # Log deduplication stats
        self.logger.info(
            "query_post_processing_complete",
            input_count=len(queries),
            output_count=len(processed),
            duplicates_removed=len(queries) - len(processed),
        )

        return processed

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text

        Returns:
            Set of lowercase keywords
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out stop words and short words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "will", "be", "is", "are", "was", "were",
            "been", "have", "has", "had", "do", "does", "did", "can", "could",
            "would", "should", "may", "might", "must", "shall", "will", "able",
        }

        keywords = {
            word for word in words
            if len(word) > 2 and word not in stop_words
        }

        return keywords

    def _has_keyword_coverage(self, query: str, question_keywords: Set[str]) -> bool:
        """
        Check if query has sufficient keyword overlap with question.

        Args:
            query: Search query
            question_keywords: Keywords from the question

        Returns:
            True if query has at least one keyword from question
        """
        query_keywords = self._extract_keywords(query)
        overlap = query_keywords & question_keywords
        return len(overlap) > 0

    def _is_too_similar(
        self,
        query: str,
        existing_queries: List[str],
        threshold: float = 0.75,
    ) -> bool:
        """
        Check if query is too similar to existing queries using fuzzy matching.

        Args:
            query: Query to check
            existing_queries: List of existing queries
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            True if query is too similar to any existing query
        """
        query_lower = query.lower()

        for existing in existing_queries:
            existing_lower = existing.lower()

            # Calculate similarity ratio
            similarity = SequenceMatcher(None, query_lower, existing_lower).ratio()

            if similarity >= threshold:
                self.logger.debug(
                    "query_similarity_check",
                    query=query,
                    existing=existing,
                    similarity=similarity,
                )
                return True

        return False


def generate_queries(
    question: str,
    run_id: int,
    prior_queries: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
) -> List[SearchQuery]:
    """
    Convenience function to generate search queries.

    Args:
        question: Forecasting question
        run_id: Run ID for database association
        prior_queries: Previously generated queries to avoid
        llm_client: Optional LLM client (defaults to GeminiClient)

    Returns:
        List of SearchQuery objects
    """
    generator = QueryGenerator(llm_client=llm_client)
    return generator.generate_queries(question, run_id, prior_queries)
