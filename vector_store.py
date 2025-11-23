import chromadb
from chromadb.config import Settings
import os
import json
from embeddings import EmbeddingGenerator

class VectorStore:
    def __init__(self, collection_name, persist_directory, embedding_config):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_config = embedding_config
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model=embedding_config["model"],
            dimensions=embedding_config.get("dimensions")
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"embedding_config": json.dumps(embedding_config)}
            )
            print(f"Created new collection: {collection_name}")
    
    def add_documents(self, chunks):
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        texts_to_embed = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            texts_to_embed.append(chunk["text"])
        
        print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        embeddings = self.embedding_generator.generate(texts_to_embed)
        
        print(f"Adding {len(ids)} documents to vector store...")
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"Successfully added {len(ids)} documents")
    
    def search(self, query, top_k=5, filters=None):
        query_embedding = self.embedding_generator.generate(query)
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        
        if filters:
            kwargs["where"] = filters
        
        results = self.collection.query(**kwargs)
        
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return retrieved_chunks
    
    def count(self):
        return self.collection.count()
    
    def reset(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"embedding_config": json.dumps(self.embedding_config)}
        )