"""
Lightweight Retrieval-Augmented Generation (RAG) utilities.

This module provides a simple implementation of RAG using numpy for vector operations.
It includes document processing, embedding, and retrieval functionality.
"""

import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict

class SimpleDocumentStore:
    """A simple document store for RAG."""
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the document store.
        
        Args:
            embedding_dim: Dimension of document embeddings
        """
        self.documents = []
        self.document_ids = []
        self.embeddings = None
        self.embedding_dim = embedding_dim
        self.metadata = []
        
    def add_document(self, document: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a document to the store.
        
        Args:
            document: Document text
            doc_id: Optional document ID
            metadata: Optional metadata for the document
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}"
            
        if metadata is None:
            metadata = {}
            
        self.documents.append(document)
        self.document_ids.append(doc_id)
        self.metadata.append(metadata)
        
        # Reset embeddings since we've added a new document
        self.embeddings = None
        
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None, 
                     metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Add multiple documents to the store.
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
            metadata_list: Optional list of metadata for the documents
        """
        if doc_ids is None:
            start_idx = len(self.documents)
            doc_ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
            
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(documents))]
            
        for doc, doc_id, metadata in zip(documents, doc_ids, metadata_list):
            self.add_document(doc, doc_id, metadata)
            
    def _compute_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute simple embeddings for texts using a basic TF-IDF-like approach.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        # Create a simple vocabulary
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)
            
        vocab = {word: i for i, word in enumerate(all_words)}
        vocab_size = len(vocab)
        
        # Use a smaller embedding dimension if vocabulary is smaller than embedding_dim
        effective_dim = min(vocab_size, self.embedding_dim)
        
        # Create a simple random projection matrix
        np.random.seed(42)  # For reproducibility
        projection = np.random.normal(0, 1, (vocab_size, effective_dim))
        
        # Compute embeddings
        embeddings = np.zeros((len(texts), effective_dim))
        
        for i, text in enumerate(texts):
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = defaultdict(int)
            
            for word in words:
                if word in vocab:
                    word_counts[vocab[word]] += 1
                    
            # Create a sparse vector
            indices = list(word_counts.keys())
            values = list(word_counts.values())
            
            if indices:
                # Apply TF-IDF-like weighting
                values = [v / len(words) * np.log(len(texts) / sum(1 for t in texts if word in t.lower())) 
                         for v, word in zip(values, [list(vocab.keys())[i] for i in indices])]
                
                # Project to embedding space
                for idx, val in zip(indices, values):
                    embeddings[i] += val * projection[idx]
                    
            # Normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
                
        # If effective_dim < embedding_dim, pad with zeros
        if effective_dim < self.embedding_dim:
            padded_embeddings = np.zeros((len(texts), self.embedding_dim))
            padded_embeddings[:, :effective_dim] = embeddings
            return padded_embeddings
            
        return embeddings
        
    def compute_embeddings(self):
        """Compute embeddings for all documents in the store."""
        if not self.documents:
            return
            
        self.embeddings = self._compute_simple_embeddings(self.documents)
        
    def save(self, path: str):
        """
        Save the document store to a file.
        
        Args:
            path: Path to save the document store
        """
        if self.embeddings is None and self.documents:
            self.compute_embeddings()
            
        data = {
            'documents': self.documents,
            'document_ids': self.document_ids,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim,
            'metadata': self.metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, path: str) -> 'SimpleDocumentStore':
        """
        Load a document store from a file.
        
        Args:
            path: Path to load the document store from
            
        Returns:
            Loaded document store
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        store = cls(embedding_dim=data['embedding_dim'])
        store.documents = data['documents']
        store.document_ids = data['document_ids']
        store.embeddings = data['embeddings']
        store.metadata = data['metadata']
        
        return store
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document information
        """
        if not self.documents:
            return []
            
        if self.embeddings is None:
            self.compute_embeddings()
            
        # Compute query embedding
        query_embedding = self._compute_simple_embeddings([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'document_id': self.document_ids[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })
            
        return results


class DocumentProcessor:
    """Process documents for RAG."""
    
    @staticmethod
    def split_text(text: str, method: str = 'paragraph', 
                  min_length: int = 50, max_length: int = 1000,
                  overlap: int = 0, **kwargs) -> List[str]:
        """
        Split text into chunks using various methods.
        
        Args:
            text: Text to split
            method: Splitting method ('paragraph', 'sentence', 'chunk', 'semantic', 'regex')
            min_length: Minimum chunk length
            max_length: Maximum chunk length
            overlap: Overlap between chunks (for 'chunk' method)
            **kwargs: Additional arguments for specific methods
            
        Returns:
            List of text chunks
        """
        if method == 'paragraph':
            # Split by double newlines
            chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if c.strip()]
            
        elif method == 'sentence':
            # Simple sentence splitting
            chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
        elif method == 'chunk':
            # Split into fixed-size chunks with overlap
            chunks = []
            for i in range(0, len(text), max_length - overlap):
                chunk = text[i:i + max_length]
                if len(chunk) >= min_length:
                    chunks.append(chunk)
                    
        elif method == 'semantic':
            # Split by headings and sections
            heading_pattern = r'(?:^|\n)#+\s+.+?(?=\n)'
            sections = re.split(heading_pattern, text)
            headings = re.findall(heading_pattern, text)
            
            chunks = []
            for i, section in enumerate(sections):
                if i > 0 and i-1 < len(headings):
                    # Prepend heading to section
                    section = headings[i-1] + section
                    
                if section.strip():
                    chunks.append(section.strip())
                    
        elif method == 'regex':
            # Split by custom regex pattern
            pattern = kwargs.get('pattern', r'\n\s*\n')
            chunks = [c.strip() for c in re.split(pattern, text) if c.strip()]
            
        else:
            # Default to paragraph splitting
            chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if c.strip()]
            
        # Filter by length
        chunks = [c for c in chunks if len(c) >= min_length]
        
        # If chunks are too long, recursively split them
        result = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                result.append(chunk)
            else:
                # Recursively split using chunk method
                result.extend(DocumentProcessor.split_text(
                    chunk, method='chunk', min_length=min_length, 
                    max_length=max_length, overlap=overlap
                ))
                
        return result


class RAGModel:
    """Retrieval-Augmented Generation model."""
    
    def __init__(self, document_store: SimpleDocumentStore, model=None):
        """
        Initialize the RAG model.
        
        Args:
            document_store: Document store for retrieval
            model: Language model for generation (optional)
        """
        self.document_store = document_store
        self.model = model
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document information
        """
        return self.document_store.search(query, top_k=top_k)
        
    def generate(self, query: str, top_k: int = 5, max_length: int = 100, 
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text based on the query and retrieved documents.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("No language model provided for generation")
            
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc['document'] for doc in retrieved_docs])
        
        # Create prompt with context and query
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response using the model
        if hasattr(self.model, 'generate'):
            # If using our custom model
            input_ids = self.model.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids]).to(next(self.model.parameters()).device)
            
            output_ids = self.model.generate(
                input_tensor, 
                max_length=max_length,
                temperature=temperature,
                top_k=int(top_p * self.model.config.get('vocab_size', 10000))
            )
            
            response = self.model.tokenizer.decode(output_ids[0].tolist())
            
        else:
            # Fallback to a simple response
            response = f"Based on the retrieved information, the answer to '{query}' would involve information from {len(retrieved_docs)} documents."
            
        return response
