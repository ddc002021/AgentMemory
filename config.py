import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
INPUT_COST_PER_1K = float(os.getenv("INPUT_COST_PER_1K", "0.0025"))
OUTPUT_COST_PER_1K = float(os.getenv("OUTPUT_COST_PER_1K", "0.01"))

WIKIPEDIA_TOPICS = [
    "Green flash",
    "Fata Morgana (mirage)",
    "Brocken spectre",
    "Circumzenithal arc",
    "Blood Falls",
    "Blood rain",
    "Lake Nyos",
    "Gravity hill",
    "Skyquake",
    "Morning Glory cloud",
    "Naga fireball",
    "Catatumbo lightning",
    "Brinicle",
    "Sailing stones"
]

SECTIONS_TO_REMOVE = [
    "References",
    "See also",
    "Further reading",
    "External links",
    "Notes",
    "Bibliography",
    "Sources"
]

VECTOR_STORE_DIR = "./data/vector_store"
CORPUS_DIR = "./data/corpus"
LTM_DB_PATH = "./data/ltm.db"
RESULTS_DIR = "./results"

STM_TOKEN_BUDGET = 2000
TOP_K_RETRIEVAL = 3

CHUNK_CONFIGS = {
    "small_fixed": {"strategy": "fixed", "size": 256, "overlap": 50},
    "large_fixed": {"strategy": "fixed", "size": 1024, "overlap": 100},
    "recursive": {"strategy": "recursive", "size": 512, "overlap": 50}
}

EMBEDDING_CONFIGS = {
    "small": {"model": "text-embedding-3-small", "dimensions": 1536},
    "large": {"model": "text-embedding-3-large", "dimensions": 3072}
}

MEMORY_CONFIGS = {
    "stm_only": {"use_stm": True, "use_ltm": False},
    "stm_ltm": {"use_stm": True, "use_ltm": True}
}

EXPERIMENT_SEED = 42