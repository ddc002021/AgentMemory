from openai import OpenAI
from config import OPENAI_API_KEY
import time

class EmbeddingGenerator:
    def __init__(self, model="text-embedding-3-small", dimensions=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.dimensions = dimensions
    
    def generate(self, texts, batch_size=100):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                kwargs = {"input": batch, "model": self.model}
                if self.dimensions:
                    kwargs["dimensions"] = self.dimensions
                
                response = self.client.embeddings.create(**kwargs)
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                raise
        
        return embeddings if len(embeddings) > 1 else embeddings[0]
    
    def get_dimensions(self):
        return self.dimensions if self.dimensions else self._get_default_dimensions()
    
    def _get_default_dimensions(self):
        if "3-small" in self.model:
            return 1536
        elif "3-large" in self.model:
            return 3072
        elif "ada-002" in self.model:
            return 1536
        else:
            return 1536