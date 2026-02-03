#!/usr/bin/env python3
"""
HSK Enrichment Server - High-performance parallel API enrichment with WebSocket streaming
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import websockets
from websockets.server import serve, WebSocketServerProtocol

# Configuration
BATCH_SIZE = 40
MAX_CONCURRENT = 15  # Parallel API requests
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
WS_HOST = "localhost"
WS_PORT = 8765


@dataclass
class BatchResult:
    batch_index: int
    success: bool
    results: List[Dict]
    error: Optional[str] = None
    duration_ms: int = 0


def build_enrichment_prompt(words: List[Dict]) -> str:
    """Build a high-quality prompt for word enrichment."""
    word_list = json.dumps(words, ensure_ascii=False)

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
    max_retries: int = 3
) -> BatchResult:
    """Call Anthropic API with retry logic and rate limit handling."""

    start_time = time.time()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                prompt = build_enrichment_prompt(words)

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
    """Process all words with parallel API calls, streaming results via WebSocket."""

    # Create batches
    batches = [words[i:i + BATCH_SIZE] for i in range(0, len(words), BATCH_SIZE)]
    total_batches = len(batches)
    total_words = len(words)

    # Send initial status
    await websocket.send(json.dumps({
        "type": "start",
        "total_words": total_words,
        "total_batches": total_batches,
        "batch_size": BATCH_SIZE,
        "concurrency": MAX_CONCURRENT
    }))

    start_time = time.time()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = 0
    success_count = 0
    failed_count = 0
    all_results = []

    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = []
        for i, batch in enumerate(batches):
            # Prepare word data for API
            word_data = [{"w": w["chinese"], "p": w.get("pinyin", ""), "l": w.get("hskLevel2026")} for w in batch]
            task = asyncio.create_task(
                call_anthropic_api(session, word_data, api_key, i, semaphore)
            )
            tasks.append((i, batch, task))

            # Send batch_queued message
            await websocket.send(json.dumps({
                "type": "batch_queued",
                "batch": i,
                "words": [w["chinese"] for w in batch[:3]] + (["..."] if len(batch) > 3 else []),
                "count": len(batch)
            }))

        # Process results as they complete
        for coro in asyncio.as_completed([t[2] for t in tasks]):
            result = await coro
            completed += 1

            if result.success:
                success_count += len(result.results)
                all_results.extend(result.results)

                await websocket.send(json.dumps({
                    "type": "batch_done",
                    "batch": result.batch_index,
                    "success": True,
                    "count": len(result.results),
                    "results": result.results,
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
                batch_words = batches[result.batch_index]
                failed_count += len(batch_words)

                await websocket.send(json.dumps({
                    "type": "batch_done",
                    "batch": result.batch_index,
                    "success": False,
                    "error": result.error,
                    "failed_words": [w["chinese"] for w in batch_words],
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
    print(f"Starting HSK Enrichment Server on ws://{WS_HOST}:{WS_PORT}")
    print(f"Configuration: batch_size={BATCH_SIZE}, max_concurrent={MAX_CONCURRENT}")

    async with serve(handle_client, WS_HOST, WS_PORT):
        print("Server ready. Waiting for connections...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
