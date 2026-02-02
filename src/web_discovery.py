"""
Web Discovery Module for NIMITZ
Uses Brave Search API to discover entities (players, people, etc.) from web searches.
"""

import os
import re
import json
import requests
from typing import List, Dict, Optional, Set
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
        self.api_key = api_key or os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise WebDiscoveryError(
                "Brave API key not provided. "
                "Set BRAVE_API_KEY environment variable or pass api_key parameter. "
                "Get a free key at: https://brave.com/search/api/"
            )
        self.base_url = "https://api.search.brave.com/res/v1"

    def search(
        self,
        query: str,
        count: int = 10,
        search_lang: str = "en"
    ) -> Dict:
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
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        params = {
            'q': query,
            'count': min(count, 20),  # Free tier max
            'search_lang': search_lang
        }

        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
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
    pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    matches = re.findall(pattern1, text)
    for match in matches:
        # Filter out common non-name patterns
        if not any(stop in match.lower() for stop in [
            'the ', 'and ', 'for ', 'with ', 'from ',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        ]):
            names.add(match)

    return names


def extract_entities_from_search_results(
    results: Dict,
    entity_type: str = "person"
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
    web_results = results.get('web', {}).get('results', [])

    for result in web_results:
        # Extract from title
        title = result.get('title', '')
        entities.update(extract_names_from_text(title))

        # Extract from description
        description = result.get('description', '')
        entities.update(extract_names_from_text(description))

    # Convert to list and sort
    return sorted(list(entities))


def discover_entities(
    query: str,
    entity_type: str = "person",
    max_results: int = 10,
    api_key: Optional[str] = None
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
    entity_type: str = "person"
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

    if file_ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(descriptions, f, indent=2)
    elif file_ext in ['.txt', '.csv', '']:
        with open(output_path, 'w') as f:
            for desc in descriptions:
                f.write(f"{desc}\n")
    else:
        raise WebDiscoveryError(f"Unsupported file format: {file_ext}")

    print(f"✓ Saved to: {output_path}")

    return entities


def interactive_entity_selection(
    entities: List[str],
    max_select: Optional[int] = None
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

        if selection == 'all':
            return entities

        if '-' in selection:
            # Range selection
            try:
                start, end = map(int, selection.split('-'))
                if 1 <= start <= len(entities) and 1 <= end <= len(entities) and start <= end:
                    return entities[start-1:end]
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
                selected = [entities[idx-1] for idx in indices]
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
    interactive: bool = True
) -> List[str]:
    """
    Discover entities with optional interactive refinement.

    Args:
        initial_query: Initial search query
        output_file: Output file path
        description_template: Optional description template
        max_results: Maximum results
        api_key: Optional Brave API key
        interactive: Enable interactive selection

    Returns:
        List of final selected entities
    """
    # Discover entities
    entities = discover_entities(initial_query, max_results=max_results * 2, api_key=api_key)

    if not entities:
        raise WebDiscoveryError(f"No entities found for: {initial_query}")

    print(f"\n✓ Found {len(entities)} potential entities")

    # Interactive selection if enabled
    if interactive and len(entities) > max_results:
        print(f"\nWould you like to review and select specific entities?")
        choice = input("Review? [Y/n]: ").strip().lower()

        if choice != 'n':
            entities = interactive_entity_selection(entities, max_select=max_results)
    else:
        entities = entities[:max_results]

    # Generate descriptions
    if description_template:
        descriptions = [description_template.format(name=name) for name in entities]
    else:
        descriptions = entities

    # Save to file
    file_ext = os.path.splitext(output_file)[1].lower()

    if file_ext == '.json':
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
    else:
        with open(output_file, 'w') as f:
            for desc in descriptions:
                f.write(f"{desc}\n")

    print(f"\n✓ Saved {len(entities)} entities to: {output_file}")

    return entities


def check_brave_api_key() -> bool:
    """Check if Brave API key is configured"""
    return bool(os.getenv('BRAVE_API_KEY'))


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
