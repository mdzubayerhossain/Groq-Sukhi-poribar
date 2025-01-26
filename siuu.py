import streamlit as st
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import re

class DocumentStore:
    def __init__(self, docs_path: str):
        """Initialize document store with Bangla documents."""
        self.docs_path = Path(docs_path)
        self.documents = {}
        self.embeddings = None
        self.encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.load_documents()
        
    def preprocess_bangla(self, text: str) -> str:
        """Preprocess Bangla text."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('।', '.')
        return text
        
    def load_documents(self):
        """Load Bangla documents from file or directory."""
        try:
            if self.docs_path.is_file():
                # Handle single file
                with open(self.docs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    processed_content = self.preprocess_bangla(content)
                    self.documents[self.docs_path.stem] = processed_content
            elif self.docs_path.is_dir():
                # Handle directory
                for file_path in self.docs_path.glob('*.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        processed_content = self.preprocess_bangla(content)
                        self.documents[file_path.stem] = processed_content
            
            if self.documents:
                texts = list(self.documents.values())
                self.embeddings = self.encoder.encode(texts, convert_to_tensor=True)
            else:
                st.error("No documents found at the specified path.")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

class SmartSearch:
    def __init__(self, document_store: DocumentStore):
        """Initialize search with document store."""
        self.document_store = document_store
        
    def search(self, query: str, top_k: int = 3) -> list:
        """Search for relevant Bangla documents based on the query."""
        try:
            if not self.document_store.embeddings is not None:
                return []
                
            processed_query = self.document_store.preprocess_bangla(query)
            query_embedding = self.document_store.encoder.encode(processed_query, convert_to_tensor=True)
            
            similarities = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                self.document_store.embeddings.cpu().numpy()
            )[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            doc_ids = list(self.document_store.documents.keys())
            for idx in top_indices:
                results.append({
                    'doc_id': doc_ids[idx],
                    'content': list(self.document_store.documents.values())[idx],
                    'score': float(similarities[idx])
                })
            
            return results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

class BanglaQABot:
    def __init__(self, docs_dir: str):
        """Initialize the Bangla QA bot with components."""
        self.document_store = DocumentStore(docs_dir)
        self.search_engine = SmartSearch(self.document_store)
        self.qa_pipeline = pipeline(
            "question-answering",
            model="sagorsarker/bangla-bert-base",
            tokenizer="sagorsarker/bangla-bert-base"
        )
        
    def answer_question(self, question: str) -> dict:
        """Process a Bangla question and generate an answer."""
        try:
            relevant_docs = self.search_engine.search(question)
            
            if not relevant_docs:
                return {
                    'question': question,
                    'answer': "দুঃখিত, কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি।",
                    'relevant_documents': []
                }
            
            context = " ".join([doc['content'] for doc in relevant_docs])
            
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            return {
                'question': question,
                'answer': result['answer'],
                'relevant_documents': relevant_docs
            }
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return {
                'question': question,
                'answer': "দুঃখিত, একটি ত্রুটি ঘটেছে।",
                'relevant_documents': []
            }

def main():
    st.title("বাংলা প্রশ্নোত্তর বট")
    
    # File uploader
    uploaded_file = st.file_uploader("আপলোড টেক্সট ফাইল", type=['txt'])
    
    # Initialize session state
    if 'qa_bot' not in st.session_state or uploaded_file:
        if uploaded_file:
            # Save uploaded file
            save_path = Path("temp_docs") / uploaded_file.name
            save_path.parent.mkdir(exist_ok=True)
            save_path.write_bytes(uploaded_file.getvalue())
            docs_path = str(save_path)
        else:
            docs_path = r"F:\Coding\project\grop\book.txt"
        
        st.session_state.qa_bot = BanglaQABot(docs_path)
    
    # Question input
    question = st.text_input("আপনার প্রশ্ন লিখুন:", key="question_input")
    
    if st.button("উত্তর পান"):
        if question:
            with st.spinner("উত্তর তৈরি করা হচ্ছে..."):
                result = st.session_state.qa_bot.answer_question(question)
                
                st.write("### উত্তর:")
                st.write(result['answer'])
                
                if result['relevant_documents']:
                    st.write("### প্রাসঙ্গিক ডকুমেন্ট:")
                    for doc in result['relevant_documents']:
                        st.write(f"- {doc['doc_id']} (সাদৃশ্য: {doc['score']:.3f})")
        else:
            st.warning("দয়া করে একটি প্রশ্ন লিখুন।")

if __name__ == "__main__":
    main()