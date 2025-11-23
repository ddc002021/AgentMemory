import tiktoken

class TextChunker:
    def __init__(self, strategy="fixed", chunk_size=512, overlap=50):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def chunk(self, text, metadata=None):
        if self.strategy == "fixed":
            return self._fixed_chunking(text, metadata)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _fixed_chunking(self, text, metadata):
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_strategy": "fixed",
                "chunk_size": self.chunk_size,
                "start_token": start,
                "end_token": end
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            start = end - self.overlap if end < len(tokens) else end
            chunk_id += 1
        
        return chunks
    
    def _recursive_chunking(self, text, metadata):
        separators = ["\n\n", "\n", ". ", " "]
        
        chunks = []
        self._recursive_split(text, separators, chunks, metadata, 0)
        return chunks
    
    def _recursive_split(self, text, separators, chunks, metadata, chunk_id_start):
        if not text.strip():
            return chunk_id_start
        
        token_count = self.count_tokens(text)
        
        if token_count <= self.chunk_size:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": chunk_id_start,
                "chunk_strategy": "recursive",
                "chunk_size": self.chunk_size
            })
            chunks.append({
                "text": text.strip(),
                "metadata": chunk_metadata
            })
            return chunk_id_start + 1
        
        if not separators:
            return self._force_split(text, chunks, metadata, chunk_id_start)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        parts = text.split(separator)
        
        current_chunk = ""
        chunk_id = chunk_id_start
        
        for i, part in enumerate(parts):
            test_chunk = current_chunk + separator + part if current_chunk else part
            
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk_id = self._recursive_split(
                        current_chunk, remaining_separators, chunks, metadata, chunk_id
                    )
                current_chunk = part
        
        if current_chunk:
            chunk_id = self._recursive_split(
                current_chunk, remaining_separators, chunks, metadata, chunk_id
            )
        
        return chunk_id
    
    def _force_split(self, text, chunks, metadata, chunk_id_start):
        tokens = self.encoding.encode(text)
        chunk_id = chunk_id_start
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_strategy": "recursive_forced",
                "chunk_size": self.chunk_size
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            chunk_id += 1
        
        return chunk_id