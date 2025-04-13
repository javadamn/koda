from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


# --------- RAG QUERY ---------
def query_graphrag(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [chunks[i] for i in indices[0]]
    combined_context = "\n---\n".join([r["text"] for r in results])
    return combined_context


# --------- RAG PIPELINE ---------


class BioRAGPipeline:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    def generate_answer(self, context: str, question: str) -> str:
        """Generate biologically structured answers"""
        structured_prompt = f"""You are a microbial systems biology assistant.
Based on the following context, answer the question below explicitly in this structured format:
Context: {context}
Question: {question}

Provide the response strictly in the following format:
1. Key Microorganisms: [List microorganisms clearly identified in the context.]
2. Metabolic Pathways: [List clearly any metabolic pathways including enzymes and genes.]
3. Cross-Feeding Relationships: [List clearly with directionality and metabolites.]
4. Ecological Impact: [Explain clearly the host health implications in 1-2 sentences.]

"""
        
        inputs = self.tokenizer(structured_prompt, 
                              return_tensors="pt", 
                              max_length=1024, 
                              truncation=True, 
                              padding='max_length')
        outputs = self.model.generate(**inputs, max_length=400, num_beams=5, early_stopping=True, no_repeat_ngram_size=3)
        return self._clean_output(outputs[0])#self._postprocess_answer(self.tokenizer.decode(outputs[0]))
    
    
    def _clean_output(self, output_tokens):
        
        return self.tokenizer.decode(
        output_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()
    
    def _postprocess_answer(self, text: str) -> str:
        """Ensure biological relevance"""
        required_terms = ['produce', 'consume', 'metabolite', 'biomass']
        if not any(term in text.lower() for term in required_terms):
            return "No significant biological relationships identified"
        return text

