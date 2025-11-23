import os
import json
from tqdm import tqdm
from data_ingestion import load_corpus
from chunking import TextChunker
from vector_store import VectorStore
from agent import Agent
from evaluation import Evaluator, load_evaluation_dataset, save_results
from config import CHUNK_CONFIGS, EMBEDDING_CONFIGS, MEMORY_CONFIGS, VECTOR_STORE_DIR, RESULTS_DIR, EXPERIMENT_SEED
import random
import numpy as np

random.seed(EXPERIMENT_SEED)
np.random.seed(EXPERIMENT_SEED)

def ingest_corpus(corpus, chunk_config, embedding_config, collection_name):
    chunker = TextChunker(
        strategy=chunk_config["strategy"],
        chunk_size=chunk_config["size"],
        overlap=chunk_config["overlap"]
    )
    
    all_chunks = []
    for article in corpus:
        metadata = {
            "title": article["title"],
            "url": article["url"]
        }
        chunks = chunker.chunk(article["content"], metadata)
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    persist_dir = os.path.join(VECTOR_STORE_DIR, collection_name)
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_config=embedding_config
    )
    
    if vector_store.count() == 0:
        vector_store.add_documents(all_chunks)
    else:
        print(f"Collection {collection_name} already has {vector_store.count()} documents")
    
    return vector_store

def run_single_experiment(config_name, chunk_config, embedding_config, memory_config, evaluation_dataset):
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}")
    
    corpus = load_corpus()
    
    collection_name = f"{config_name}_{chunk_config['strategy']}_{chunk_config['size']}"
    
    vector_store = ingest_corpus(corpus, chunk_config, embedding_config, collection_name)
    
    agent = Agent(vector_store=vector_store, use_stm=memory_config["use_stm"], use_ltm=memory_config["use_ltm"])
    
    evaluator = Evaluator()
    
    # Determine chunking key for ground truth lookup
    chunking_key = f"{chunk_config['strategy']}_{chunk_config['size']}"
    
    results = []
    
    for item in tqdm(evaluation_dataset, desc="Evaluating queries"):
        question = item["question"]
        reference_answer = item["reference_answer"]
        
        # Get gold chunk IDs for this specific chunking strategy
        gold_chunk_ids = item.get("gold_chunk_ids", {}).get(chunking_key, [])
        
        response = agent.answer(question)
        
        retrieval_metrics = evaluator.evaluate_retrieval_quality(
            response["retrieved_chunks"],
            gold_chunk_ids
        )
        
        answer_metrics = evaluator.evaluate_answer_quality(
            response["answer"],
            reference_answer
        )
        
        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": response["answer"],
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "latency": response["latency"],
            "tokens_in": response["tokens_in"],
            "tokens_out": response["tokens_out"]
        })
        
        agent.reset_session()
    
    aggregated_metrics = evaluator.aggregate_metrics(results)
    
    experiment_result = {
        "config_name": config_name,
        "chunk_config": chunk_config,
        "embedding_config": embedding_config,
        "memory_config": memory_config,
        "results": results,
        "aggregated_metrics": aggregated_metrics
    }
    
    return experiment_result

def run_all_experiments():
    evaluation_dataset = load_evaluation_dataset()
    
    if not evaluation_dataset:
        print("No evaluation dataset found. Please create evaluation_dataset.json first.")
        return
    
    print(f"Loaded {len(evaluation_dataset)} evaluation queries")
    
    all_experiment_results = []
    
    print("\n" + "="*60)
    print("EXPERIMENT A: Chunk Size Comparison")
    print("="*60)
    
    for chunk_name in ["small_fixed", "large_fixed"]:
        config_name = f"exp_a_{chunk_name}"
        result = run_single_experiment(
            config_name=config_name,
            chunk_config=CHUNK_CONFIGS[chunk_name],
            embedding_config=EMBEDDING_CONFIGS["small"],
            memory_config=MEMORY_CONFIGS["stm_ltm"],
            evaluation_dataset=evaluation_dataset
        )
        all_experiment_results.append(result)
    
    print("\n" + "="*60)
    print("EXPERIMENT B: Chunking Strategy Comparison")
    print("="*60)
    
    for chunk_name in ["small_fixed", "recursive"]:
        config_name = f"exp_b_{chunk_name}"
        result = run_single_experiment(
            config_name=config_name,
            chunk_config=CHUNK_CONFIGS[chunk_name],
            embedding_config=EMBEDDING_CONFIGS["small"],
            memory_config=MEMORY_CONFIGS["stm_ltm"],
            evaluation_dataset=evaluation_dataset
        )
        all_experiment_results.append(result)
    
    print("\n" + "="*60)
    print("EXPERIMENT C: Embedding Model Comparison")
    print("="*60)
    
    for embedding_name in ["small", "large"]:
        config_name = f"exp_c_{embedding_name}"
        result = run_single_experiment(
            config_name=config_name,
            chunk_config=CHUNK_CONFIGS["small_fixed"],
            embedding_config=EMBEDDING_CONFIGS[embedding_name],
            memory_config=MEMORY_CONFIGS["stm_ltm"],
            evaluation_dataset=evaluation_dataset
        )
        all_experiment_results.append(result)
    
    print("\n" + "="*60)
    print("EXPERIMENT D: Memory Policy Comparison")
    print("="*60)
    
    for memory_name in ["stm_only", "stm_ltm"]:
        config_name = f"exp_d_{memory_name}"
        result = run_single_experiment(
            config_name=config_name,
            chunk_config=CHUNK_CONFIGS["small_fixed"],
            embedding_config=EMBEDDING_CONFIGS["small"],
            memory_config=MEMORY_CONFIGS[memory_name],
            evaluation_dataset=evaluation_dataset
        )
        all_experiment_results.append(result)
    
    results_file = os.path.join(RESULTS_DIR, "all_experiments.json")
    save_results(all_experiment_results, results_file)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)
    print(f"Results saved to: {results_file}")
    
    return all_experiment_results

if __name__ == "__main__":
    results = run_all_experiments()