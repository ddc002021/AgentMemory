import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import RESULTS_DIR

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_chunk_size_comparison(results, output_dir):
    exp_a_results = [r for r in results if r["config_name"].startswith("exp_a_")]
    
    if len(exp_a_results) < 2:
        return
    
    data = []
    for result in exp_a_results:
        chunk_size = result["chunk_config"]["size"]
        metrics = result["aggregated_metrics"]
        
        data.append({
            "Chunk Size": chunk_size,
            "Hit Rate": metrics["retrieval"]["hit_rate_mean"],
            "MRR": metrics["retrieval"]["mrr_mean"],
            "Semantic Similarity": metrics["answer_quality"]["semantic_similarity_mean"],
            "Latency (s)": metrics["latency"]["p50"],
            "Cost ($)": metrics["cost"]["estimated_total_usd"] / len(result["results"])
        })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment A: Chunk Size Comparison", fontsize=16, fontweight='bold')
    
    df.plot(x="Chunk Size", y="Hit Rate", ax=axes[0, 0], marker='o', legend=False)
    axes[0, 0].set_title("Hit Rate @ k")
    axes[0, 0].set_ylabel("Hit Rate")
    axes[0, 0].grid(True, alpha=0.3)
    
    df.plot(x="Chunk Size", y="Semantic Similarity", ax=axes[0, 1], marker='o', legend=False, color='green')
    axes[0, 1].set_title("Answer Quality")
    axes[0, 1].set_ylabel("Semantic Similarity")
    axes[0, 1].grid(True, alpha=0.3)
    
    df.plot(x="Chunk Size", y="Latency (s)", ax=axes[1, 0], marker='o', legend=False, color='red')
    axes[1, 0].set_title("Latency")
    axes[1, 0].set_ylabel("Latency (seconds)")
    axes[1, 0].grid(True, alpha=0.3)
    
    df.plot(x="Chunk Size", y="Cost ($)", ax=axes[1, 1], marker='o', legend=False, color='purple')
    axes[1, 1].set_title("Cost per Query")
    axes[1, 1].set_ylabel("Cost (USD)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp_a_chunk_size.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_chunking_strategy_comparison(results, output_dir):
    exp_b_results = [r for r in results if r["config_name"].startswith("exp_b_")]
    
    if len(exp_b_results) < 2:
        return
    
    strategies = []
    hit_rates = []
    similarities = []
    latencies = []
    
    for result in exp_b_results:
        strategy = result["chunk_config"]["strategy"]
        metrics = result["aggregated_metrics"]
        
        strategies.append(strategy)
        hit_rates.append(metrics["retrieval"]["hit_rate_mean"])
        similarities.append(metrics["answer_quality"]["semantic_similarity_mean"])
        latencies.append(metrics["latency"]["p50"])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Experiment B: Chunking Strategy Comparison", fontsize=16, fontweight='bold')
    
    axes[0].bar(strategies, hit_rates, color=['skyblue', 'lightcoral'])
    axes[0].set_title("Hit Rate @ k")
    axes[0].set_ylabel("Hit Rate")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(strategies, similarities, color=['lightgreen', 'gold'])
    axes[1].set_title("Answer Quality")
    axes[1].set_ylabel("Semantic Similarity")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(strategies, latencies, color=['plum', 'peachpuff'])
    axes[2].set_title("Latency")
    axes[2].set_ylabel("Latency (seconds)")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp_b_chunking_strategy.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_embedding_comparison(results, output_dir):
    exp_c_results = [r for r in results if r["config_name"].startswith("exp_c_")]
    
    if len(exp_c_results) < 2:
        return
    
    models = []
    hit_rates = []
    similarities = []
    costs = []
    
    for result in exp_c_results:
        model = result["embedding_config"]["model"].split("-")[-1]
        metrics = result["aggregated_metrics"]
        
        models.append(model)
        hit_rates.append(metrics["retrieval"]["hit_rate_mean"])
        similarities.append(metrics["answer_quality"]["semantic_similarity_mean"])
        costs.append(metrics["cost"]["estimated_total_usd"] / len(result["results"]))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Experiment C: Embedding Model Comparison", fontsize=16, fontweight='bold')
    
    axes[0].bar(models, hit_rates, color=['steelblue', 'darkorange'])
    axes[0].set_title("Hit Rate @ k")
    axes[0].set_ylabel("Hit Rate")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(models, similarities, color=['mediumseagreen', 'indianred'])
    axes[1].set_title("Answer Quality")
    axes[1].set_ylabel("Semantic Similarity")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(models, costs, color=['mediumpurple', 'gold'])
    axes[2].set_title("Cost per Query")
    axes[2].set_ylabel("Cost (USD)")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp_c_embedding.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_policy_comparison(results, output_dir):
    exp_d_results = [r for r in results if r["config_name"].startswith("exp_d_")]
    
    if len(exp_d_results) < 2:
        return
    
    policies = []
    hit_rates = []
    similarities = []
    latencies = []
    
    for result in exp_d_results:
        if result["memory_config"]["use_ltm"]:
            policy = "STM + LTM"
        else:
            policy = "STM Only"
        
        metrics = result["aggregated_metrics"]
        
        policies.append(policy)
        hit_rates.append(metrics["retrieval"]["hit_rate_mean"])
        similarities.append(metrics["answer_quality"]["semantic_similarity_mean"])
        latencies.append(metrics["latency"]["p50"])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Experiment D: Memory Policy Comparison", fontsize=16, fontweight='bold')
    
    axes[0].bar(policies, hit_rates, color=['teal', 'coral'])
    axes[0].set_title("Hit Rate @ k")
    axes[0].set_ylabel("Hit Rate")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(policies, similarities, color=['olive', 'pink'])
    axes[1].set_title("Answer Quality")
    axes[1].set_ylabel("Semantic Similarity")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(policies, latencies, color=['slateblue', 'sandybrown'])
    axes[2].set_title("Latency")
    axes[2].set_ylabel("Latency (seconds)")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp_d_memory_policy.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(results, output_dir):
    data = []
    
    for result in results:
        metrics = result["aggregated_metrics"]
        
        data.append({
            "Experiment": result["config_name"],
            "Hit Rate": f"{metrics['retrieval']['hit_rate_mean']:.3f}",
            "MRR": f"{metrics['retrieval']['mrr_mean']:.3f}",
            "Similarity": f"{metrics['answer_quality']['semantic_similarity_mean']:.3f}",
            "Latency (s)": f"{metrics['latency']['p50']:.3f}",
            "Cost ($)": f"{metrics['cost']['estimated_total_usd'] / len(result['results']):.6f}"
        })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, len(data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title("Summary of All Experiments", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_results():
    results_file = os.path.join(RESULTS_DIR, "all_experiments.json")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    results = load_results(results_file)
    
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    plot_chunk_size_comparison(results, plots_dir)
    print("Generated: Chunk Size Comparison")
    
    plot_chunking_strategy_comparison(results, plots_dir)
    print("Generated: Chunking Strategy Comparison")
    
    plot_embedding_comparison(results, plots_dir)
    print("Generated: Embedding Model Comparison")
    
    plot_memory_policy_comparison(results, plots_dir)
    print("Generated: Memory Policy Comparison")
    
    generate_summary_table(results, plots_dir)
    print("Generated: Summary Table")
    
    print(f"\nAll plots saved to: {plots_dir}")

if __name__ == "__main__":
    visualize_all_results()