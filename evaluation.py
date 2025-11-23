import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from config import INPUT_COST_PER_1K, OUTPUT_COST_PER_1K

class Evaluator:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_retrieval_quality(self, retrieved_chunks, gold_chunk_ids, k=5):
        """
        Evaluate retrieval quality at chunk-level using ID-based matching.
        
        Args:
            retrieved_chunks: List of retrieved chunks with 'id' and 'metadata'
            gold_chunk_ids: List of chunk IDs that are relevant for this chunking strategy
            k: Number of top results to consider
        
        Returns:
            Dictionary with hit_rate and mrr
        """
        if not gold_chunk_ids:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "retrieved_count": len(retrieved_chunks)
            }
        
        retrieved_ids = [chunk["id"] for chunk in retrieved_chunks[:k]]
        gold_set = set(gold_chunk_ids)
        
        # Find positions of relevant chunks
        relevant_positions = []
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in gold_set:
                relevant_positions.append(i)
        
        # Hit rate: Did we retrieve at least one relevant chunk?
        hit_rate = 1.0 if len(relevant_positions) > 0 else 0.0
        
        # MRR: Position of first relevant chunk
        mrr = 1.0 / (relevant_positions[0] + 1) if relevant_positions else 0.0
        
        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "retrieved_count": len(retrieved_chunks)
        }
    
    def evaluate_answer_quality(self, generated_answer, reference_answer):
        if not reference_answer or not generated_answer:
            return {"semantic_similarity": 0.0}
        
        gen_embedding = self.similarity_model.encode([generated_answer])
        ref_embedding = self.similarity_model.encode([reference_answer])
        
        similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
        
        return {
            "semantic_similarity": float(similarity)
        }
    
    def aggregate_metrics(self, results):
        if not results:
            return {}
        
        hit_rates = [r["retrieval_metrics"]["hit_rate"] for r in results]
        mrrs = [r["retrieval_metrics"]["mrr"] for r in results]
        similarities = [r["answer_metrics"]["semantic_similarity"] for r in results]
        latencies = [r["latency"] for r in results]
        tokens_in = [int(r["tokens_in"]) for r in results]
        tokens_out = [int(r["tokens_out"]) for r in results]
        
        aggregated = {
            "retrieval": {
                "hit_rate_mean": np.mean(hit_rates),
                "mrr_mean": np.mean(mrrs)
            },
            "answer_quality": {
                "semantic_similarity_mean": np.mean(similarities)
            },
            "latency": {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95)
            },
            "cost": {
                "tokens_in_total": sum(tokens_in),
                "tokens_out_total": sum(tokens_out)
            }
        }
        
        total_cost = (
            (aggregated["cost"]["tokens_in_total"] / 1000) * INPUT_COST_PER_1K +
            (aggregated["cost"]["tokens_out_total"] / 1000) * OUTPUT_COST_PER_1K
        )
        
        aggregated["cost"]["estimated_total_usd"] = total_cost
        
        return aggregated

def load_evaluation_dataset(filepath="evaluation_dataset.json"):
    if not os.path.exists(filepath):
        print(f"Evaluation dataset not found at {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")