"""
Web Discovery Module for NIMITZ
Uses Brave Search API to discover entities (players, people, etc.) from web searches.
"""

import os
import re
import json
import requests
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlencode


class WebDiscoveryError(Exception):
    """Base exception for web discovery errors"""

    pass


class BraveSearchClient:
    """Client for Brave Search API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave Search API key. If None, uses BRAVE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise WebDiscoveryError(
                "Brave API key not provided. "
                "Set BRAVE_API_KEY environment variable or pass api_key parameter. "
                "Get a free key at: https://brave.com/search/api/"
            )
        self.base_url = "https://api.search.brave.com/res/v1"

    def search(self, query: str, count: int = 10, search_lang: str = "en") -> Dict:
        """
        Perform a web search.

        Args:
            query: Search query
            count: Number of results (max 20 for free tier)
            search_lang: Search language code

        Returns:
            Dict with search results

        Raises:
            WebDiscoveryError if search fails
        """
        endpoint = f"{self.base_url}/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": min(count, 20),  # Free tier max
            "search_lang": search_lang,
        }

        try:
            response = requests.get(
                endpoint, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise WebDiscoveryError(f"Brave Search API error: {e}")


def extract_names_from_text(text: str) -> Set[str]:
    """
    Extract potential names from text using simple heuristics.

    Args:
        text: Text to extract names from

    Returns:
        Set of potential names (deduplicated)
    """
    names = set()

    # Pattern 1: Capitalized words (First Last, First Middle Last)
    # Match 2-4 capitalized words in sequence
    pattern1 = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
    matches = re.findall(pattern1, text)
    for match in matches:
        # Filter out common non-name patterns
        if not any(
            stop in match.lower()
            for stop in [
                "the ",
                "and ",
                "for ",
                "with ",
                "from ",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
        ):
            names.add(match)

    return names


def extract_entities_from_search_results(
    results: Dict, entity_type: str = "person"
) -> List[str]:
    """
    Extract entities (names, people, etc.) from Brave Search results.

    Args:
        results: Brave Search API results
        entity_type: Type of entity to extract ('person', 'organization', etc.)

    Returns:
        List of extracted entities
    """
    entities = set()

    # Extract from web results
    web_results = results.get("web", {}).get("results", [])

    for result in web_results:
        # Extract from title
        title = result.get("title", "")
        entities.update(extract_names_from_text(title))

        # Extract from description
        description = result.get("description", "")
        entities.update(extract_names_from_text(description))

    # Convert to list and sort
    return sorted(list(entities))


def discover_entities(
    query: str,
    entity_type: str = "person",
    max_results: int = 10,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Discover entities (people, players, etc.) from a web search.

    Args:
        query: Search query (e.g., "Parma Clima Baseball team roster")
        entity_type: Type of entity to discover
        max_results: Maximum number of results to return
        api_key: Optional Brave API key

    Returns:
        List of discovered entity names

    Raises:
        WebDiscoveryError if discovery fails
    """
    client = BraveSearchClient(api_key=api_key)

    # Enhance query for better results
    if entity_type == "person":
        enhanced_query = f"{query} players roster members"
    else:
        enhanced_query = query

    # Perform search
    results = client.search(enhanced_query, count=10)

    # Extract entities
    entities = extract_entities_from_search_results(results, entity_type)

    # Return top N results
    return entities[:max_results]


def discover_and_generate_batch_file(
    query: str,
    output_file: str,
    description_template: Optional[str] = None,
    max_results: int = 20,
    api_key: Optional[str] = None,
    entity_type: str = "person",
) -> List[str]:
    """
    Discover entities and generate a batch file for image retrieval.

    Args:
        query: Search query
        output_file: Path to output file (.txt, .json, .csv)
        description_template: Optional template for descriptions (e.g., "{name}, baseball player")
        max_results: Maximum entities to discover
        api_key: Optional Brave API key
        entity_type: Type of entity to discover

    Returns:
        List of discovered entities

    Raises:
        WebDiscoveryError if discovery fails
    """
    # Discover entities
    print(f"🔍 Searching: {query}")
    entities = discover_entities(query, entity_type, max_results, api_key)

    if not entities:
        raise WebDiscoveryError(f"No entities found for query: {query}")

    print(f"✓ Found {len(entities)} entities")

    # Generate descriptions
    if description_template:
        descriptions = [description_template.format(name=name) for name in entities]
    else:
        descriptions = entities

    # Save to file
    output_path = output_file
    file_ext = os.path.splitext(output_path)[1].lower()

    if file_ext == ".json":
        with open(output_path, "w") as f:
            json.dump(descriptions, f, indent=2)
    elif file_ext in [".txt", ".csv", ""]:
        with open(output_path, "w") as f:
            for desc in descriptions:
                f.write(f"{desc}\n")
    else:
        raise WebDiscoveryError(f"Unsupported file format: {file_ext}")

    print(f"✓ Saved to: {output_path}")

    return entities


def interactive_entity_selection(
    entities: List[str], max_select: Optional[int] = None
) -> List[str]:
    """
    Interactive selection of entities from a list.

    Args:
        entities: List of discovered entities
        max_select: Optional maximum number to select

    Returns:
        List of selected entities
    """
    print(f"\n📋 Found {len(entities)} entities:\n")

    for i, entity in enumerate(entities, 1):
        print(f"  {i:2d}. {entity}")

    print("\n" + "=" * 60)
    print("Select entities to include:")
    print("  - Enter numbers separated by spaces (e.g., '1 3 5')")
    print("  - Enter 'all' to select all")
    print("  - Enter 'range' for a range (e.g., '1-10')")
    print("=" * 60)

    while True:
        selection = input("\nSelection: ").strip().lower()

        if selection == "all":
            return entities

        if "-" in selection:
            # Range selection
            try:
                start, end = map(int, selection.split("-"))
                if (
                    1 <= start <= len(entities)
                    and 1 <= end <= len(entities)
                    and start <= end
                ):
                    return entities[start - 1 : end]
                else:
                    print(f"Invalid range. Use 1-{len(entities)}")
                    continue
            except ValueError:
                print("Invalid range format. Use: start-end (e.g., '1-10')")
                continue

        # Individual selection
        try:
            indices = [int(x) for x in selection.split()]
            if all(1 <= idx <= len(entities) for idx in indices):
                selected = [entities[idx - 1] for idx in indices]
                if max_select and len(selected) > max_select:
                    print(f"Too many selected. Maximum: {max_select}")
                    continue
                return selected
            else:
                print(f"Invalid selection. Use numbers 1-{len(entities)}")
        except ValueError:
            print("Invalid input. Enter numbers, 'all', or a range")


def discover_with_refinement(
    initial_query: str,
    output_file: str,
    description_template: Optional[str] = None,
    max_results: int = 20,
    api_key: Optional[str] = None,
    interactive: bool = True,
    use_llm_filter: bool = True,
    llm_provider: str = "auto",
) -> List[str]:
    """
    Discover entities with optional interactive refinement and LLM filtering.

    Args:
        initial_query: Initial search query
        output_file: Output file path
        description_template: Optional description template
        max_results: Maximum results
        api_key: Optional Brave API key
        interactive: Enable interactive selection
        use_llm_filter: Use LLM to filter person names vs organizations
        llm_provider: LLM provider for filtering

    Returns:
        List of final selected entities
    """
    # Discover entities
    entities = discover_entities(
        initial_query, max_results=max_results * 3, api_key=api_key
    )

    if not entities:
        raise WebDiscoveryError(f"No entities found for: {initial_query}")

    print(f"\n✓ Found {len(entities)} potential entities")

    # Interactive filtering if enabled
    if interactive:
        # Extract context from query for better LLM filtering
        context = initial_query
        entities = interactive_filter_confirmation(
            entities, context=context, use_llm=use_llm_filter, llm_provider=llm_provider
        )
    elif use_llm_filter:
        # Non-interactive but still use LLM to filter
        try:
            print("🤖 Using LLM to filter person names...")
            person_names, filtered_out = filter_names_with_llm(
                entities, context=initial_query, provider=llm_provider
            )
            print(f"✓ Filtered to {len(person_names)} person names")
            entities = person_names
        except Exception as e:
            print(f"⚠️  LLM filtering failed: {e}")
            print("   Using all entities")

    # Limit to max_results
    if len(entities) > max_results:
        entities = entities[:max_results]

    # Generate descriptions
    if description_template:
        descriptions = [description_template.format(name=name) for name in entities]
    else:
        descriptions = entities

    # Save to file
    file_ext = os.path.splitext(output_file)[1].lower()

    if file_ext == ".json":
        with open(output_file, "w") as f:
            json.dump(descriptions, f, indent=2)
    else:
        with open(output_file, "w") as f:
            for desc in descriptions:
                f.write(f"{desc}\n")

    print(f"\n✓ Saved {len(entities)} entities to: {output_file}")

    return entities


def check_brave_api_key() -> bool:
    """Check if Brave API key is configured"""
    return bool(os.getenv("BRAVE_API_KEY"))


def get_brave_api_info() -> str:
    """Get information about Brave Search API"""
    return """
Brave Search API - Free Tier Available

Get your free API key at: https://brave.com/search/api/

Free tier includes:
- 2,000 queries per month
- Up to 20 results per query
- Web search, news, and more

Once you have your key, set it as an environment variable:
  export BRAVE_API_KEY="your-api-key-here"
"""


def filter_names_with_llm(
    entities: List[str], context: str = "baseball players", provider: str = "auto"
) -> Tuple[List[str], List[str]]:
    """
    Use LLM to filter entity list, separating person names from organizations/noise.

    Args:
        entities: List of potential entity names to filter
        context: Context hint (e.g., "baseball players", "artists")
        provider: LLM provider to use ("auto", "anthropic", "gemini", "openai")

    Returns:
        Tuple of (person_names, filtered_out) lists

    Raises:
        ImportError if litellm not installed
        ValueError if no LLM API key configured
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm not installed. Install with: pip install litellm\n"
            "Required for LLM-based name filtering."
        )

    # Import LLM config from llm_analyzer
    try:
        from llm_analyzer import get_llm_config
    except ImportError:
        raise ImportError("Could not import llm_analyzer module")

    # Get LLM configuration
    config = get_llm_config(provider)

    # Prepare prompt
    prompt = f"""You are analyzing a list of entities extracted from web search results about "{context}".

Your task: Classify each entity as either:
- "person": A real person's name (first + last name)
- "other": Organization, team, generic term, or not a person name

Entities to classify:
{json.dumps(entities, indent=2)}

Return ONLY a JSON object with this exact structure:
{{
  "person_names": ["Name One", "Name Two", ...],
  "filtered_out": ["Filtered One", "Filtered Two", ...]
}}

Rules:
1. "person_names" must contain ONLY individual human names
2. Exclude organizations (e.g., "Baseball Almanac", "New York Yankees")
3. Exclude generic terms (e.g., "Complete List", "Current Players")
4. Exclude job titles/positions (e.g., "First Base", "Pitcher")
5. Include only names you're confident are real people
6. If unsure, put it in "filtered_out"

Return ONLY the JSON, no other text."""

    try:
        # Prepare litellm arguments
        llm_args = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistent classification
            "max_tokens": 2048,
        }

        # Add custom base_url if provided (for vLLM or other OpenAI-compatible endpoints)
        if config.base_url:
            llm_args["api_base"] = config.base_url

        # Add API key if provided
        if config.api_key:
            llm_args["api_key"] = config.api_key

        # Call LLM
        response = litellm.completion(**llm_args)

        # Extract response
        content = response.choices[0].message.content.strip()

        # Try to parse JSON (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        person_names = result.get("person_names", [])
        filtered_out = result.get("filtered_out", [])

        return person_names, filtered_out

    except Exception as e:
        print(f"⚠️  LLM filtering failed: {e}")
        print(f"   Falling back to original list")
        return entities, []


def interactive_filter_confirmation(
    entities: List[str],
    context: str = "entities",
    use_llm: bool = True,
    llm_provider: str = "auto",
) -> List[str]:
    """
    Interactively filter entities with optional LLM assistance.

    Args:
        entities: List of entities to filter
        context: Context for filtering (e.g., "baseball players")
        use_llm: Whether to use LLM for automatic filtering
        llm_provider: LLM provider to use

    Returns:
        Filtered list of entities
    """
    print(f"\n{'=' * 60}")
    print(f"  Entity Filtering - {len(entities)} entities found")
    print(f"{'=' * 60}\n")

    person_names = []
    filtered_out = []

    # Try LLM filtering first
    if use_llm:
        try:
            print("🤖 Using LLM to filter results...")
            person_names, filtered_out = filter_names_with_llm(
                entities, context, llm_provider
            )
            print(f"✓ LLM classified {len(person_names)} as person names")
            print(f"  Filtered out {len(filtered_out)} non-person entities\n")

        except (ImportError, ValueError) as e:
            print(f"⚠️  LLM filtering unavailable: {e}")
            print(f"   Continuing with manual review...\n")
            person_names = entities
            use_llm = False

    else:
        person_names = entities

    # Show preview
    print("📋 Person names found:")
    print("-" * 60)
    for i, name in enumerate(person_names[:20], 1):
        print(f"  {i:2d}. {name}")
    if len(person_names) > 20:
        print(f"  ... and {len(person_names) - 20} more")
    print()

    if filtered_out and use_llm:
        print("🗑️  Filtered out (organizations/noise):")
        print("-" * 60)
        for i, name in enumerate(filtered_out[:10], 1):
            print(f"  {i:2d}. {name}")
        if len(filtered_out) > 10:
            print(f"  ... and {len(filtered_out) - 10} more")
        print()

    # Ask for confirmation
    print(f"{'=' * 60}")
    print("Options:")
    print("  1. Use these person names (recommended)")
    print("  2. Use ALL entities (including filtered out)")
    print("  3. Manual selection")
    print("  4. Cancel")
    print(f"{'=' * 60}")

    while True:
        choice = input("\nChoice [1-4]: ").strip()

        if choice == "1":
            return person_names
        elif choice == "2":
            return entities
        elif choice == "3":
            return interactive_entity_selection(person_names)
        elif choice == "4":
            raise WebDiscoveryError("Discovery cancelled by user")
        else:
            print("Invalid choice. Enter 1, 2, 3, or 4")
