"""Central configuration. Every tunable number lives here so experiments are easy to reproduce."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SAMPLES_DIR = DATA_DIR / "samples"
DEEPLINKS_PATH = DATA_DIR / "deeplinks.json"
SIIS_PATH = DATA_DIR / "siis_responses.json"
INPUT_PATH = DATA_DIR / "input.txt"

# ---- Output rules from the problem statement (section 4.1) ----
TITLE_MIN_WORDS, TITLE_MAX_WORDS = 2, 3
DESC_MIN_WORDS, DESC_MAX_WORDS = 5, 7
VARIATIONS_MIN, VARIATIONS_MAX = 8, 10
GOAL_KINDS = ("Troubleshooting", "Configuration")

# ---- Grounding: a step must share this fraction of its content words with the source ----
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.6"))

# ---- Below this relevance the engine answers "no_match" instead of guessing ----
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.3"))                    # LLM-judged relevance
RELEVANCE_MIN_LEXICAL = float(os.getenv("RELEVANCE_MIN_LEXICAL", "0.05"))   # offline word-overlap relevance


@dataclass
class LLMSettings:
    # "openai" = any OpenAI-compatible API (OpenAI, Groq, Gemini's OpenAI endpoint, Ollama...)
    # "offline" = deterministic rule-based extractor, no API key needed
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or None)
    temperature: float = 0.0
    seed: int | None = field(default_factory=lambda: int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else 42)
    max_repair_attempts: int = 2
    timeout_s: float = 30.0
    # USD per 1M tokens, used for cost tracking. Update for the model you use.
    price_in_per_m: float = field(default_factory=lambda: float(os.getenv("LLM_PRICE_IN", "0.15")))
    price_out_per_m: float = field(default_factory=lambda: float(os.getenv("LLM_PRICE_OUT", "0.60")))

    @property
    def use_offline(self) -> bool:
        return self.provider == "offline" or not self.api_key


# =====================================================================  Phase 2
INDEX_DIR = DATA_DIR / "index"

# "st"    = sentence-transformers dense embeddings (best quality, downloads a ~130 MB model once)
# "tfidf" = character n-gram TF-IDF (no download, lighter, used as fallback)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "st")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

RETRIEVAL_TOP_K = 15          # candidates pulled from the hybrid index per action
RRF_K = 60                    # reciprocal-rank-fusion constant
# a catalog entry is accepted only if its setting name covers a screen named in the steps (see name_match)
MAP_MIN_NAME_MATCH = float(os.getenv("MAP_MIN_NAME_MATCH", "0.6"))
DUMMY_DEEPLINK = "voiceassist://dummy_positive"

# =====================================================================  Phase 3
CACHE_DB_PATH = DATA_DIR / "cache" / "plans.db"
# a cached plan is reused when a new query is at least this similar to one of its stored phrasings...
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.85"))
# ...AND beats the best DIFFERENT plan by this margin (stops near-ties returning the wrong plan)
CACHE_MARGIN = float(os.getenv("CACHE_MARGIN", "0.02"))
HELDOUT_PATH = DATA_DIR / "eval" / "heldout_paraphrases.json"