import wikipedia
import os
import json
import re
from config import WIKIPEDIA_TOPICS, SECTIONS_TO_REMOVE, CORPUS_DIR

def clean_article_content(content):
    for section in SECTIONS_TO_REMOVE:
        if f"== {section} ==" in content:
            content = content.split(f"== {section} ==")[0]
        
    return content

def fetch_wikipedia_articles():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    
    articles = []
    
    for topic in WIKIPEDIA_TOPICS:
        try:
            print(f"Fetching: {topic}")
            page = wikipedia.page(topic, auto_suggest=False)
            
            content = clean_article_content(page.content)
            
            article_data = {
                "title": page.title,
                "url": page.url,
                "content": content,
                "summary": page.summary
            }
            
            articles.append(article_data)
            
            filename = f"{topic.replace(' ', '_').replace('(', '').replace(')', '')}.json"
            filepath = os.path.join(CORPUS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
    
    print(f"\nSuccessfully fetched {len(articles)} articles")
    return articles

def load_corpus():
    articles = []
    
    if not os.path.exists(CORPUS_DIR):
        print("Corpus directory not found. Fetching articles...")
        return fetch_wikipedia_articles()
    
    # CRITICAL: Sort filenames to ensure deterministic chunk ID generation
    filenames = sorted(os.listdir(CORPUS_DIR))
    for filename in filenames:
        if filename.endswith('.json'):
            filepath = os.path.join(CORPUS_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                articles.append(json.load(f))
    
    if not articles:
        print("No articles found. Fetching articles...")
        return fetch_wikipedia_articles()
    
    return articles

if __name__ == "__main__":
    articles = fetch_wikipedia_articles()
    print(f"\nCorpus ready with {len(articles)} articles")