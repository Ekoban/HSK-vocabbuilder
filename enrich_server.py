#!/usr/bin/env python3
"""
HSK Enrichment Server - High-performance parallel API enrichment with WebSocket streaming
Uses CC-CEDICT as primary source for Chinese→English translations.
LLM is only called for: (1) words not in dictionary, (2) French translations.
"""

import asyncio
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import websockets
from websockets.server import serve, WebSocketServerProtocol

# Configuration
BATCH_SIZE = 40
MAX_CONCURRENT = 30  # Parallel API requests
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
WS_HOST = "localhost"
WS_PORT = 8765
CEDICT_PATH = Path(__file__).parent / "cedict.txt"

# Global dictionary cache
CEDICT: Dict[str, List[Dict[str, str]]] = {}  # simplified -> [{trad, pinyin, definitions}, ...]


def parse_cedict_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse a CC-CEDICT line. Returns (traditional, simplified, pinyin, definitions) or None."""
    if not line or line.startswith('#'):
        return None

    # Format: Traditional Simplified [pinyin] /def1/def2/.../
    match = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/$', line)
    if not match:
        return None

    trad, simp, pinyin, defs = match.groups()
    return trad, simp, pinyin, defs


def load_cedict() -> int:
    """Load CC-CEDICT into memory. Returns number of entries loaded."""
    global CEDICT
    CEDICT.clear()

    if not CEDICT_PATH.exists():
        print(f"Warning: CC-CEDICT not found at {CEDICT_PATH}")
        return 0

    count = 0
    with open(CEDICT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_cedict_line(line.strip())
            if parsed:
                trad, simp, pinyin, defs = parsed
                entry = {
                    'traditional': trad,
                    'pinyin': pinyin,
                    'definitions': defs
                }
                # Index by simplified (primary) and traditional (fallback)
                if simp not in CEDICT:
                    CEDICT[simp] = []
                CEDICT[simp].append(entry)

                if trad != simp and trad not in CEDICT:
                    CEDICT[trad] = []
                if trad != simp:
                    CEDICT[trad].append(entry)

                count += 1

    print(f"Loaded {count:,} CC-CEDICT entries ({len(CEDICT):,} unique keys)")
    return count


def lookup_cedict(word: str) -> Optional[Dict[str, str]]:
    """Look up a word in CC-CEDICT. Returns best match or None."""
    entries = CEDICT.get(word)
    if not entries:
        return None

    # Return first entry (most common usage)
    entry = entries[0]

    # Clean up definitions: take first few, remove technical markers
    defs = entry['definitions'].split('/')
    # Filter out pronunciation variants, see also refs, etc.
    clean_defs = []
    for d in defs[:3]:  # Take up to 3 definitions
        d = d.strip()
        if d and not d.startswith('see ') and not d.startswith('also ') and not d.startswith('variant of'):
            # Remove classifier markers like (Tw), (PRC), etc.
            d = re.sub(r'\s*\([^)]*\)\s*', ' ', d).strip()
            if d:
                clean_defs.append(d)

    if not clean_defs:
        # Fall back to raw first definition
        clean_defs = [defs[0].strip()] if defs else []

    return {
        'traditional': entry['traditional'],
        'pinyin_cedict': entry['pinyin'],
        'en': clean_defs[0] if clean_defs else None,
        'en_all': '; '.join(clean_defs[:3]) if len(clean_defs) > 1 else None
    }


@dataclass
class BatchResult:
    batch_index: int
    success: bool
    results: List[Dict]
    error: Optional[str] = None
    duration_ms: int = 0


def build_enrichment_prompt(words: List[Dict], mode: str = "full") -> str:
    """Build a prompt for word enrichment.

    Args:
        words: List of word dicts with 'w' (Chinese), 'p' (pinyin), 'l' (HSK level), and optionally 'en' (English from CC-CEDICT)
        mode: "full" = need everything, "french_only" = have English from CC-CEDICT, just need French + metadata
    """
    word_list = json.dumps(words, ensure_ascii=False)

    if mode == "french_only":
        # Words with CC-CEDICT English - just need French translation + metadata
        return f"""You are a Chinese-French lexicographer. For each word below, provide a French translation and metadata.
Each word includes its CC-CEDICT English translation - use this as reference for the French translation.

CRITICAL FOR FRENCH:
- Use natural, idiomatic French - NOT literal translations from English
- Example: 外卖 (takeout) → "plats à emporter" (NOT "livraison à domicile")
- Example: 加油 (to cheer on) → "bon courage" or "allez" (NOT "ajouter de l'essence")

Return a JSON array with these exact fields for each word:
- w: the Chinese word (echo back exactly)
- fr: concise French translation (1-4 words, idiomatic French)
- ipa: IPA transcription with Mandarin tone markers (˥ ˧˥ ˨˩˦ ˥˩ for tones 1-4) in /slashes/
- cat: grammatical category (noun, verb, adjective, adverb, measure word, pronoun, conjunction, preposition, particle, interjection, numeral, phrase, idiom)
- topic: semantic field - pick ONE: greeting, family, food, drink, body, health, clothing, home, school, work, transport, shopping, money, weather, time, nature, animal, color, number, technology, sport, travel, culture, emotion, social, government, science, media, law, abstract, daily_life
- sc: total stroke count (sum of all characters)
- freq: usage frequency 1-5 (5=very common daily words, 1=rare/specialized)
- diff: difficulty for learners 1-5 (1=HSK1-2 level, 5=advanced/literary)

RESPOND WITH ONLY THE JSON ARRAY. No markdown, no explanation.

Words ({len(words)}):
{word_list}"""

    else:
        # Full enrichment for words not in CC-CEDICT
        return f"""You are a professional Chinese-English-French lexicographer. For each Chinese word below, provide precise, natural translations and linguistic metadata.

CRITICAL REQUIREMENTS FOR TRANSLATIONS:
- English: Use the most common, natural translation. For verbs, use infinitive form. For nouns, use singular unless typically plural.
- French: Use natural French, NOT literal translations from English. Consider French usage patterns.
  - Example: 外卖 → "plats à emporter" (NOT "livraison à domicile" which is delivery service)
  - Example: 加油 → "bon courage" or "allez" (NOT "ajouter de l'essence" unless context is gas station)
- IPA: Use standard IPA with Mandarin tone markers (˥ ˧˥ ˨˩˦ ˥˩ for tones 1-4)

Return a JSON array with these exact fields for each word:
- w: the Chinese word (echo back exactly as provided)
- en: concise English translation (1-4 words, natural usage)
- fr: concise French translation (1-4 words, idiomatic French)
- ipa: IPA transcription with tone diacritics in /slashes/
- cat: grammatical category (noun, verb, adjective, adverb, measure word, pronoun, conjunction, preposition, particle, interjection, numeral, phrase, idiom)
- topic: semantic field - pick ONE: greeting, family, food, drink, body, health, clothing, home, school, work, transport, shopping, money, weather, time, nature, animal, color, number, technology, sport, travel, culture, emotion, social, government, science, media, law, abstract, daily_life
- sc: total stroke count (sum of all characters)
- freq: usage frequency 1-5 (5=very common daily words, 1=rare/specialized)
- diff: difficulty for learners 1-5 (1=HSK1-2 level, 5=advanced/literary)

RESPOND WITH ONLY THE JSON ARRAY. No markdown, no explanation.

Words to process ({len(words)} words):
{word_list}"""


async def call_anthropic_api(
    session: aiohttp.ClientSession,
    words: List[Dict],
    api_key: str,
    batch_index: int,
    semaphore: asyncio.Semaphore,
    mode: str = "full",
    max_retries: int = 3
) -> BatchResult:
    """Call Anthropic API with retry logic and rate limit handling."""

    start_time = time.time()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                prompt = build_enrichment_prompt(words, mode)

                async with session.post(
                    ANTHROPIC_API,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 4096,
                        "system": "You are a Chinese dictionary API. Return ONLY valid JSON arrays. No markdown formatting.",
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:

                    if resp.status == 429:  # Rate limited
                        retry_after = int(resp.headers.get("retry-after", 2 ** attempt))
                        await asyncio.sleep(retry_after + 0.5)
                        continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"API error {resp.status}: {error_text[:200]}")

                    data = await resp.json()
                    text = "".join(c.get("text", "") for c in data.get("content", []))

                    # Parse JSON response
                    clean_text = text.strip()
                    if clean_text.startswith("```"):
                        clean_text = clean_text.split("```")[1]
                        if clean_text.startswith("json"):
                            clean_text = clean_text[4:]
                        clean_text = clean_text.strip()

                    results = json.loads(clean_text)

                    if not isinstance(results, list):
                        raise Exception("Response is not a JSON array")

                    # Validate and clean results
                    validated = []
                    for item in results:
                        if item and isinstance(item, dict) and item.get("w"):
                            validated.append({
                                "w": str(item["w"]),
                                "en": item.get("en"),
                                "fr": item.get("fr"),
                                "ipa": item.get("ipa"),
                                "cat": item.get("cat"),
                                "topic": item.get("topic"),
                                "sc": item.get("sc") if isinstance(item.get("sc"), int) else None,
                                "freq": item.get("freq") if isinstance(item.get("freq"), int) else None,
                                "diff": item.get("diff") if isinstance(item.get("diff"), int) else None,
                            })

                    duration_ms = int((time.time() - start_time) * 1000)
                    return BatchResult(
                        batch_index=batch_index,
                        success=True,
                        results=validated,
                        duration_ms=duration_ms
                    )

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    results=[],
                    error="Request timed out",
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            except json.JSONDecodeError as e:
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    results=[],
                    error=f"Invalid JSON response: {str(e)}",
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    results=[],
                    error=str(e),
                    duration_ms=int((time.time() - start_time) * 1000)
                )

    return BatchResult(
        batch_index=batch_index,
        success=False,
        results=[],
        error="Max retries exceeded",
        duration_ms=int((time.time() - start_time) * 1000)
    )


async def process_enrichment(
    websocket: WebSocketServerProtocol,
    words: List[Dict],
    api_key: str
):
    """Process all words with parallel API calls, streaming results via WebSocket.

    Uses CC-CEDICT as primary source for English translations.
    LLM is called for: (1) words not in CC-CEDICT, (2) French translations + metadata.
    """

    total_words = len(words)
    start_time = time.time()

    # Phase 1: Look up all words in CC-CEDICT
    cedict_hits = {}  # chinese -> cedict data
    cedict_misses = []  # words not in dictionary

    for w in words:
        chinese = w["chinese"]
        lookup = lookup_cedict(chinese)
        if lookup and lookup.get('en'):
            cedict_hits[chinese] = lookup
        else:
            cedict_misses.append(w)

    found_count = len(cedict_hits)
    miss_count = len(cedict_misses)

    # Send initial status with CC-CEDICT stats
    await websocket.send(json.dumps({
        "type": "start",
        "total_words": total_words,
        "cedict_found": found_count,
        "cedict_missing": miss_count,
        "batch_size": BATCH_SIZE,
        "concurrency": MAX_CONCURRENT,
        "message": f"CC-CEDICT: {found_count} found, {miss_count} need full LLM enrichment"
    }))

    # Prepare batches - all words go to LLM but with different modes
    # Words in CC-CEDICT: french_only mode (we already have English)
    # Words not in CC-CEDICT: full mode (need English too)

    # Group words by mode for efficient batching
    french_only_words = [w for w in words if w["chinese"] in cedict_hits]
    full_enrich_words = cedict_misses

    all_batches = []
    batch_index = 0

    # Create french_only batches
    for i in range(0, len(french_only_words), BATCH_SIZE):
        batch = french_only_words[i:i + BATCH_SIZE]
        word_data = []
        for w in batch:
            chinese = w["chinese"]
            cedict_data = cedict_hits[chinese]
            word_data.append({
                "w": chinese,
                "p": w.get("pinyin", ""),
                "l": w.get("hskLevel2026"),
                "en": cedict_data["en"]  # Include CC-CEDICT English for context
            })
        all_batches.append({
            "index": batch_index,
            "mode": "french_only",
            "words": batch,
            "word_data": word_data
        })
        batch_index += 1

    # Create full enrichment batches
    for i in range(0, len(full_enrich_words), BATCH_SIZE):
        batch = full_enrich_words[i:i + BATCH_SIZE]
        word_data = [{"w": w["chinese"], "p": w.get("pinyin", ""), "l": w.get("hskLevel2026")} for w in batch]
        all_batches.append({
            "index": batch_index,
            "mode": "full",
            "words": batch,
            "word_data": word_data
        })
        batch_index += 1

    total_batches = len(all_batches)

    if total_batches == 0:
        await websocket.send(json.dumps({
            "type": "complete",
            "total_words": total_words,
            "success_words": 0,
            "failed_words": 0,
            "total_time_ms": int((time.time() - start_time) * 1000),
            "words_per_second": 0,
            "all_results": []
        }))
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = 0
    success_count = 0
    failed_count = 0
    all_results = []

    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = []
        for batch_info in all_batches:
            task = asyncio.create_task(
                call_anthropic_api(
                    session,
                    batch_info["word_data"],
                    api_key,
                    batch_info["index"],
                    semaphore,
                    mode=batch_info["mode"]
                )
            )
            tasks.append((batch_info, task))

            # Send batch_queued message
            mode_label = "📖 CEDICT+French" if batch_info["mode"] == "french_only" else "🤖 Full LLM"
            await websocket.send(json.dumps({
                "type": "batch_queued",
                "batch": batch_info["index"],
                "mode": batch_info["mode"],
                "words": [w["chinese"] for w in batch_info["words"][:3]] + (["..."] if len(batch_info["words"]) > 3 else []),
                "count": len(batch_info["words"]),
                "message": f"{mode_label}: {len(batch_info['words'])} words"
            }))

        # Process results as they complete
        for coro in asyncio.as_completed([t[1] for t in tasks]):
            result = await coro
            completed += 1

            # Find the batch info for this result
            batch_info = next(b for b, _ in tasks if b["index"] == result.batch_index)

            if result.success:
                # Merge CC-CEDICT English with LLM results for french_only batches
                merged_results = []
                for llm_result in result.results:
                    word = llm_result.get("w")
                    if batch_info["mode"] == "french_only" and word in cedict_hits:
                        # Use CC-CEDICT English, LLM French + metadata
                        cedict_data = cedict_hits[word]
                        merged = {
                            "w": word,
                            "en": cedict_data["en"],  # From CC-CEDICT
                            "en_source": "cedict",
                            "fr": llm_result.get("fr"),
                            "ipa": llm_result.get("ipa"),
                            "cat": llm_result.get("cat"),
                            "topic": llm_result.get("topic"),
                            "sc": llm_result.get("sc"),
                            "freq": llm_result.get("freq"),
                            "diff": llm_result.get("diff"),
                        }
                        # Include traditional if different
                        if cedict_data.get("traditional") and cedict_data["traditional"] != word:
                            merged["trad"] = cedict_data["traditional"]
                        merged_results.append(merged)
                    else:
                        # Full LLM result
                        llm_result["en_source"] = "llm"
                        merged_results.append(llm_result)

                success_count += len(merged_results)
                all_results.extend(merged_results)

                await websocket.send(json.dumps({
                    "type": "batch_done",
                    "batch": result.batch_index,
                    "mode": batch_info["mode"],
                    "success": True,
                    "count": len(merged_results),
                    "results": merged_results,
                    "duration_ms": result.duration_ms,
                    "progress": {
                        "completed": completed,
                        "total": total_batches,
                        "success_words": success_count,
                        "failed_words": failed_count,
                        "elapsed_ms": int((time.time() - start_time) * 1000)
                    }
                }))
            else:
                # Count failed words
                failed_count += len(batch_info["words"])

                await websocket.send(json.dumps({
                    "type": "batch_done",
                    "batch": result.batch_index,
                    "mode": batch_info["mode"],
                    "success": False,
                    "error": result.error,
                    "failed_words": [w["chinese"] for w in batch_info["words"]],
                    "duration_ms": result.duration_ms,
                    "progress": {
                        "completed": completed,
                        "total": total_batches,
                        "success_words": success_count,
                        "failed_words": failed_count,
                        "elapsed_ms": int((time.time() - start_time) * 1000)
                    }
                }))

    # Send completion message
    total_time = time.time() - start_time
    await websocket.send(json.dumps({
        "type": "complete",
        "total_words": total_words,
        "success_words": success_count,
        "failed_words": failed_count,
        "cedict_used": found_count,
        "llm_full": miss_count,
        "total_time_ms": int(total_time * 1000),
        "words_per_second": round(success_count / total_time, 1) if total_time > 0 else 0,
        "all_results": all_results
    }))


async def handle_client(websocket: WebSocketServerProtocol):
    """Handle a WebSocket client connection."""
    print(f"Client connected from {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

                elif msg_type == "enrich":
                    words = data.get("words", [])
                    api_key = data.get("api_key", "")

                    if not words:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "No words provided"
                        }))
                        continue

                    if not api_key:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "No API key provided"
                        }))
                        continue

                    await process_enrichment(websocket, words, api_key)

                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}"
                    }))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON message"
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")


async def main():
    """Start the WebSocket server."""
    print("=" * 60)
    print("HSK Enrichment Server")
    print("=" * 60)

    # Load CC-CEDICT dictionary
    print("\nLoading CC-CEDICT dictionary...")
    entry_count = load_cedict()
    if entry_count == 0:
        print("WARNING: CC-CEDICT not loaded. All words will use full LLM enrichment.")
        print(f"Download CC-CEDICT to: {CEDICT_PATH}")
    else:
        print(f"CC-CEDICT ready: {entry_count:,} entries")

    print(f"\nStarting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    print(f"Configuration: batch_size={BATCH_SIZE}, max_concurrent={MAX_CONCURRENT}")

    async with serve(handle_client, WS_HOST, WS_PORT):
        print("\nServer ready. Waiting for connections...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
