import argparse
from data_ingestion import fetch_wikipedia_articles
from run_experiments import run_all_experiments
from visualize_results import visualize_all_results

def main():
    parser = argparse.ArgumentParser(description="Agent Memory System with RAG")
    parser.add_argument(
        "--mode",
        choices=["fetch", "experiment", "visualize", "all"],
        default="all",
        help="Mode to run: fetch articles, run experiments, visualize results, or all"
    )
    
    args = parser.parse_args()
    
    if args.mode == "fetch" or args.mode == "all":
        print("\n" + "="*60)
        print("FETCHING WIKIPEDIA ARTICLES")
        print("="*60)
        fetch_wikipedia_articles()
    
    if args.mode == "experiment" or args.mode == "all":
        print("\n" + "="*60)
        print("RUNNING EXPERIMENTS")
        print("="*60)
        run_all_experiments()
    
    if args.mode == "visualize" or args.mode == "all":
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        visualize_all_results()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()