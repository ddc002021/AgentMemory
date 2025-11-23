from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K_RETRIEVAL
from memory import ShortTermMemory, LongTermMemory, EntityExtractor
import time

class Agent:
    def __init__(self, vector_store, use_stm=True, use_ltm=True):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.vector_store = vector_store
        self.use_stm = use_stm
        self.use_ltm = use_ltm
        
        self.stm = ShortTermMemory() if use_stm else None
        self.ltm = LongTermMemory() if use_ltm else None
        self.entity_extractor = EntityExtractor() if use_ltm else None
    
    def answer(self, question, top_k=TOP_K_RETRIEVAL):
        start_time = time.time()
        
        retrieved_chunks = self.vector_store.search(question, top_k=top_k)
        
        context = self._build_context(retrieved_chunks)
        
        prompt = self._build_prompt(question, context)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        
        if self.use_stm:
            self.stm.add_message("user", question)
            self.stm.add_message("assistant", answer)
        
        if self.use_ltm:
            self._update_ltm(question, answer, retrieved_chunks)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency": latency
        }
    
    def _build_context(self, retrieved_chunks):
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("title", "Unknown")
            context_parts.append(f"[Source {i+1}: {source}]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question, rag_context):
        prompt_parts = []
        
        prompt_parts.append("You are a knowledgeable assistant specializing in unusual natural phenomena.")
        
        if self.use_stm and self.stm.messages:
            conversation_history = self.stm.get_context_string()
            prompt_parts.append(f"\nConversation History:\n{conversation_history}\n")
        
        if self.use_ltm:
            ltm_facts = self.ltm.get_facts(limit=5, min_salience=0.3)
            if ltm_facts:
                facts_text = "\n".join([f"- {fact['content']}" for fact in ltm_facts])
                prompt_parts.append(f"\nRelevant Facts from Previous Sessions:\n{facts_text}\n")
            
            entities = self.ltm.get_all_entities()
            if entities:
                entities_text = "\n".join([
                    f"- {e['name']} ({e['type']})" for e in entities[:10]
                ])
                prompt_parts.append(f"\nKnown Entities:\n{entities_text}\n")
            
            # Add entity relationships to provide richer context
            relations = self.ltm.get_entity_relations(limit=15)
            if relations:
                relations_text = "\n".join([
                    f"- {r['entity1']} {r['relation_type'].replace('_', ' ')} {r['entity2']}"
                    for r in relations
                ])
                prompt_parts.append(f"\nKnown Relationships:\n{relations_text}\n")
        
        prompt_parts.append(f"\nRelevant Context:\n{rag_context}\n")
        prompt_parts.append(f"\nQuestion: {question}\n")
        prompt_parts.append("\nProvide a clear and accurate answer based on the context provided.")
        
        return "\n".join(prompt_parts)
    
    def _update_ltm(self, question, answer, retrieved_chunks):
        if not retrieved_chunks:
            return
        
        best_chunk = retrieved_chunks[0]
        source = best_chunk["metadata"].get("title", "Unknown")
        
        salience = 0.7 if len(retrieved_chunks) > 0 else 0.5
        
        fact_content = f"Q: {question[:100]}... A: {answer[:200]}..."
        self.ltm.save_fact(
            content=fact_content,
            source=source,
            salience=salience,
            success_outcome=True
        )
        
        # Extract entities from question, answer, and retrieved chunks
        combined_text = question + " " + answer
        # Also include the best retrieved chunk for more context
        if retrieved_chunks:
            combined_text += " " + retrieved_chunks[0]["text"][:500]
        
        entities = self.entity_extractor.extract_from_text(combined_text)
        
        # Save all entities first
        for entity in entities:
            self.ltm.save_entity(
                name=entity["name"],
                entity_type=entity["type"],
                attributes=entity["attributes"]
            )
        
        # Extract and save relationships
        relationships = self.entity_extractor.extract_relationships(combined_text, entities)
        
        for entity1_name, entity2_name, relation_type in relationships:
            relation_id = self.ltm.save_relation(entity1_name, entity2_name, relation_type)
            if relation_id:
                # Optionally save a fact about this relationship
                self.ltm.save_fact(
                    content=f"{entity1_name} {relation_type} {entity2_name}",
                    source=source,
                    salience=0.6,
                    success_outcome=True
                )
    
    def reset_session(self):
        if self.stm:
            self.stm.clear()
    
    def get_stm_context(self):
        if self.stm:
            return self.stm.get_context_string()
        return ""
    
    def get_ltm_stats(self):
        if self.ltm:
            facts = self.ltm.get_facts(limit=100)
            entities = self.ltm.get_all_entities()
            relations = self.ltm.get_entity_relations(limit=100)
            return {
                "total_facts": len(facts),
                "total_entities": len(entities),
                "total_relations": len(relations)
            }
        return {"total_facts": 0, "total_entities": 0, "total_relations": 0}