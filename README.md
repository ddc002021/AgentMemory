Dany Chahine - 202107582

EECE798S - Assignment 6

# Agent Memory System with RAG

A stateful LLM agent implementation with short-term and long-term memory capabilities, using Retrieval-Augmented Generation (RAG) to answer complex questions about unusual natural phenomena.

---
## Architecture:

#### <u>Core Components</u>

**Memory System**
- **Short-Term Memory (STM)**: Token-budgeted (2000 tokens) rolling buffer that maintains conversation context within a session. Uses tiktoken to count tokens and automatically trims older messages when the budget is exceeded.
- **Long-Term Memory (LTM)**: SQLite database with three tables:
  - `facts`: Stores question-answer pairs with salience scores (0.5-0.7) and success outcomes
  - `entities`: Tracks phenomena, locations, and people with JSON attributes
  - `entity_relations`: Records relationships between entities (e.g., "occurs_in", "studied")
- **Entity Extractor**: Rule-based pattern matching using regex to identify:
  - Natural phenomena (14 predefined patterns)
  - Locations (via preposition-based extraction)
  - People (via capitalized name patterns)
  - Relationships between entities

**RAG Pipeline**
- **Text Chunking**: 
  - Fixed: Hard token boundaries with configurable overlap (256/1024 tokens)
  - Recursive: Semantic splitting on paragraphs -> sentences -> tokens (512 tokens)
- **Embeddings**: OpenAI API with two model options:
  - text-embedding-3-small (1536 dimensions, lower cost)
  - text-embedding-3-large (3072 dimensions, higher quality)
- **Vector Store**: Chroma with persistent storage and metadata filtering

**Agent**
- Retrieves top-k chunks from vector store based on query similarity
- Constructs dynamic prompts with:
  - System instructions
  - STM conversation history (if enabled)
  - LTM facts and entity relationships (if enabled)
  - Retrieved RAG context with source attribution
- Generates answers via GPT-4o with temperature 0.7
- Updates memory after each interaction

#### <u>When to Use Short-Term vs. Long-Term Memory</u>

**Short-Term Memory (STM) is Sufficient When:**
- Single-session, focused conversations with immediate context needs
- Questions that don't require cross-session knowledge retention

**Long-Term Memory (LTM) is Essential When:**
- Multi-session continuity where entities and facts must persist
- Complex domain knowledge requiring relationship tracking
- Entity-centric queries (e.g., "What phenomena occur in Antarctica?")

#### <u>Vector Store Choice: Chroma</u>

**Why Chroma?**
- **FAISS**: 10-100x faster for million-scale datasets but requires manual serialization and lacks metadata filtering. Overkill for our ~1500 chunk corpus.
- **Qdrant**: Production-grade with advanced filtering and distributed deployment, but adds Docker/server complexity which is unnecessary for this assignment.
- **Chroma**: Sweet spot for our assignment (persistent, metadata-aware, and low overhead).

---

## Dataset

This project uses **14 Wikipedia articles** about unusual natural phenomena as the knowledge corpus. All articles were retrieved from Wikipedia.

**Articles included:**
1. **Green flash** - Brief green light visible at sunset/sunrise
2. **Fata Morgana (mirage)** - Complex form of superior mirage
3. **Brocken spectre** - Magnified shadow of observer on clouds
4. **Circumzenithal arc** - Ice halo appearing as upside-down rainbow
5. **Blood Falls** - Red waterfall in Antarctica from iron-rich brine
6. **Blood rain** - Rain colored by airborne particles or algae spores
7. **Lake Nyos** - Crater lake with deadly CO₂ eruption history
8. **Gravity hill** - Optical illusion where downhill appears uphill
9. **Skyquake** - Unexplained booming sounds from the sky
10. **Morning Glory cloud** - Rare roll cloud formation in Australia
11. **Naga fireball** - Glowing orbs rising from Mekong River
12. **Catatumbo lightning** - Continuous lightning over Lake Maracaibo
13. **Brinicle** - Underwater ice stalactite formation
14. **Sailing stones** - Rocks that mysteriously move across desert floor

**Source:** Wikipedia (https://en.wikipedia.org)  

---

## Setup:

#### Prerequisites
- Python 3.10 or 3.11 (project developed and tested with Python 3.11)
- OpenAI API key

#### Installation

```bash
pip install -r requirements.txt
```

#### Configuration

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-2024-08-06
INPUT_COST_PER_1K=0.0025
OUTPUT_COST_PER_1K=0.01
```

---

## Usage

#### ⚠️ CRITICAL: Do NOT Run Fetch Mode

**The corpus is pre-committed to preserve chunk ID determinism.**

Running `python main.py --mode fetch` will:
- Re-download Wikipedia articles with potentially different content
- Generate new chunk IDs that won't match `evaluation_dataset.json`
- Break all gold chunk ID references used for retrieval evaluation

**Only run experiments or visualizations:**

```bash
# Run experiments with existing corpus
python main.py --mode experiment

# Generate visualizations from results
python main.py --mode visualize
```

#### Full Pipeline (Skip Fetch)

If you want to run everything except fetch:

```bash
# Manually run experiment and visualize
python main.py --mode experiment
python main.py --mode visualize
```

---

## Experiments

#### <u>A. Chunk Size Comparison</u>

**Configurations:**
- Small: 256 tokens with 50 token overlap
- Large: 1024 tokens with 100 token overlap
- Fixed chunking strategy, text-embedding-3-small, STM+LTM

**Hypothesis:** Larger chunks provide more context per retrieval, improving answer quality but reducing granularity.

**Semantic Similarity Reasoning:**
- **Small chunks (256)**: High granularity captures specific facts but may fragment semantic units (such as splitting multi-sentence explanations)
- **Large chunks (1024)**: Preserve complete explanations and cross-references but dilute semantic focus, potentially matching broader but less precise queries

#### <u>B. Chunking Strategy Comparison</u>

**Configurations:**
- Fixed (256): Hard boundaries at token limits with 50 token overlap
- Recursive (512): Semantic splitting on `\n\n` -> `\n` -> `. ` -> ` ` with 50 token overlap
- text-embedding-3-small, STM+LTM

**Hypothesis:** Recursive chunking respects semantic boundaries, improving retrieval quality by avoiding mid-sentence splits.

**Semantic Similarity Reasoning:**
- **Fixed chunking**: Deterministic and fast but may split sentences/paragraphs, creating chunks with incomplete semantic units
- **Recursive chunking**: Respects natural language structure (paragraphs -> sentences) to create coherent chunks, theoretically improving embedding quality
- **Trade-off**: Recursive chunks have variable sizes, which may introduce noise in similarity scores for very short or long chunks

#### <u>C. Embedding Model Comparison</u>

**Configurations:**
- Small: text-embedding-3-small (1536 dimensions)
- Large: text-embedding-3-large (3072 dimensions)
- Fixed chunking (256 tokens), STM+LTM

**Hypothesis:** Higher-dimensional embeddings capture richer semantics, improving retrieval quality for nuanced queries.

**Semantic Similarity Reasoning:**
- **text-embedding-3-small**: 1536 dimensions, optimized for cost-performance balance. Sufficient for well-defined factual queries but may struggle with subtle semantic distinctions.
- **text-embedding-3-large**: 3072 dimensions (2x capacity), captures finer semantic nuances
- **Expected Benefit**: Most pronounced for complex, multi-part queries requiring semantic disambiguation

#### <u>D. Memory Policy Comparison</u>

**Configurations:**
- STM Only: Conversation context within 2000 token budget
- STM + LTM: Conversation context + persistent facts, entities, and relationships
- Fixed chunking (256 tokens), text-embedding-3-small

**Hypothesis:** LTM improves answer quality by injecting entity knowledge and relationships into prompts.

**Impact on Semantic Similarity:**
- **STM-only**: Retrieval driven purely by query-chunk similarity
- **STM+LTM**: Adds entity context to prompts (e.g., "known entities: Blood Falls (phenomenon), Antarctica (location)"), potentially improving answer coherence
- **Trade-off**: LTM overhead increases latency and cost with minimal single-turn accuracy benefit (see results below)

---

## Evaluation Metrics

**Retrieval Quality**
- Hit Rate @ k
- Mean Reciprocal Rank (MRR)

**Answer Quality**
- Semantic Similarity

**Performance**
- Latency
- Cost

---

## Experimental Results

#### Summary Table

| Experiment | Hit Rate@3 | MRR | Similarity | Latency (s) | Cost ($) |
|------------|-----------|-----|------------|-------------|----------|
| **A. Small Fixed (256)** | 0.833 | 0.676 | 0.839 | 8.336 | 0.005977 |
| **A. Large Fixed (1024)** | **1.000** | **0.870** | **0.843** | 6.081 | 0.010895 |
| **B. Small Fixed (256)** | 0.833 | 0.676 | **0.841** | 5.925 | 0.005945 |
| **B. Recursive (512)** | 0.833 | 0.741 | 0.819 | 7.563 | 0.007048 |
| **C. Small Embedding** | 0.833 | 0.676 | 0.828 | 7.538 | 0.006088 |
| **C. Large Embedding** | 0.889 | 0.796 | 0.836 | 8.669 | 0.006185 |
| **D. STM Only** | 0.833 | 0.676 | 0.838 | **4.462** | **0.004823** |
| **D. STM + LTM** | 0.833 | 0.676 | 0.833 | 4.697 | 0.005995 |

---

## Key Findings

#### <u>Experiment A: Chunk Size</u>
**Winner: Large Fixed (1024 tokens)**

- **Insight**: Larger chunks preserve semantic completeness, improving retrieval at the cost of higher API usage

#### <u>Experiment B: Chunking Strategy</u>
**Winner: Small Fixed (256 tokens) by narrow margin**

- **Insight**: Recursive chunking improves ranking but fixed chunking produces better final answers. Recursive's semantic boundaries don't justify the latency cost for this corpus.

#### <u>Experiment C: Embedding Model</u>
**Winner: Large Embedding (3072d) for retrieval, Small for cost**

- **Insight**: text-embedding-3-large captures semantic nuances better, but text-embedding-3-small offers 98% of the quality at lower cost/latency

#### <u>Experiment D: Memory Policy</u>
**Winner: STM Only for single-turn QA**

- **Insight**: LTM provides no benefit for isolated queries. Use STM+LTM for multi-turn conversations requiring entity tracking.

---

## Overall Best Configuration

**For single-turn accuracy**: Large Fixed (1024) + Small Embedding + STM+LTM
- Maximizes hit rate (1.000) and semantic similarity (0.843)
- Cost: $0.0109 per query

**For cost-performance balance**: Small Fixed (256) + Small Embedding + STM Only
- Strong accuracy (0.833 hit rate, 0.838 similarity)
- Fast (4.46s) and cheap ($0.0048 per query)
- **Recommended for production** with modest compute budgets

---

## Project Structure

```
.
├── config.py                 # Configuration: API keys, chunk/embedding configs, constants
├── data_ingestion.py         # Wikipedia fetching (⚠️ DON'T RUN - breaks chunk IDs)
├── chunking.py               # TextChunker: fixed and recursive strategies
├── embeddings.py             # EmbeddingGenerator: OpenAI API wrapper
├── vector_store.py           # VectorStore: Chroma client with metadata filtering
├── memory.py                 # ShortTermMemory, LongTermMemory, EntityExtractor
├── agent.py                  # Agent: RAG pipeline + memory integration
├── evaluation.py             # Evaluator: hit-rate, MRR, semantic similarity
├── run_experiments.py        # Orchestrates 8 experiments (A-D)
├── visualize_results.py      # Generates comparison plots
├── main.py                   # CLI: --mode {experiment, visualize}
├── evaluation_dataset.json   # 18 questions with gold chunk IDs
├── requirements.txt          # Dependencies: openai, chromadb, sentence-transformers
├── README.md                 # This file
└── .env.example              # example of environment variables

data/
├── corpus/                   # ⚠️ Pre-committed articles (14 JSON files)
│   ├── Blood_Falls.json
│   ├── Green_flash.json
│   └── ... (12 more)
├── vector_store/             # Chroma collections (8 subdirectories)
│   ├── exp_a_small_fixed_fixed_256/
│   ├── exp_a_large_fixed_fixed_1024/
│   └── ... (6 more experiments)
└── ltm.db                    # SQLite: facts, entities, entity_relations tables

results/
├── all_experiments.json      # Full results: per-query metrics + aggregates
└── plots/                    # Visualizations (5 PNG files)
    ├── exp_a_chunk_size.png
    ├── exp_b_chunking_strategy.png
    ├── exp_c_embedding.png
    ├── exp_d_memory_policy.png
    └── summary_table.png
```

### Memory Policy

**STM Configuration:**
- 2000 token budget supports 4-6 conversation turns
- Uses tiktoken with `cl100k_base` encoding for accurate GPT-4 token counting
- FIFO eviction: Oldest messages removed when budget exceeded

**LTM Write Policy:**
- **Facts**: Save Q&A pair after every query with salience 0.7
- **Entities**: Extract from question + answer + top retrieved chunk (first 500 chars)
- **Relationships**: Pattern-based extraction for "occurs_in" and "studied" relations
- **Success flagging**: All completed queries marked as successful

**LTM Read Policy:**
- Retrieve top 5 facts with salience ≥ 0.3
- Include up to 10 entities and 15 relationships in prompt
- Injected after conversation history, before RAG context

**When to Use:**
- **STM-only**: Single-turn QA, cost-sensitive apps, evaluation baselines
- **STM+LTM**: Multi-session agents, entity-centric queries, knowledge graph construction