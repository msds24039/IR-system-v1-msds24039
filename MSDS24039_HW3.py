#!/usr/bin/env python
# coding: utf-8

# *Information Retrieval IR system*

# In[1]:


# Customized function for statistical insights of dataset
import re
from collections import Counter

class CorpusAnalyzer:
    def __init__(self, documents):
        self.documents = documents
        self.stats = {
            "hyphens": Counter(),
            "apostrophes": Counter(),
            "acronyms": Counter(),
            "colons": Counter(),
            "numbers": Counter()
        }

    def analyze(self):
        print("Analyzing corpus for special token patterns...")
        
        for doc in self.documents:
            text = f"{doc['heading']} {doc['content']}"
            
            # Hyphenated words 
            hyphens = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b', text)
            self.stats["hyphens"].update(hyphens)
            
            # Apostrophes
            apostrophes = re.findall(r'\b[a-zA-Z]+\'[a-zA-Z]+\b', text)
            self.stats["apostrophes"].update(apostrophes)
            
            # Acronyms / Abbreviations with dots
            acronyms = re.findall(r'\b(?:[A-Z]\.)+[A-Z]?\b', text)
            self.stats["acronyms"].update(acronyms)
            
            # Colons (often used in news for "KARACHI: ...")
            colons = re.findall(r'\b[A-Z][a-zA-Z]*:', text)
            self.stats["colons"].update(colons)

    def print_report(self):
        print("\n=== CORPUS ANALYSIS REPORT ===")
        
        print(f"\n[Hyphenated Words] Top 10 of {len(self.stats['hyphens'])} unique:")
        for w, c in self.stats['hyphens'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Apostrophes] Top 10 of {len(self.stats['apostrophes'])} unique:")
        for w, c in self.stats['apostrophes'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Acronyms] Top 10 of {len(self.stats['acronyms'])} unique:")
        for w, c in self.stats['acronyms'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Colons/Headers] Top 10:")
        for w, c in self.stats['colons'].most_common(10):
            print(f"  {w}: {c}")

        print("\n=== RECOMMENDATION ===")
        self._generate_recommendation()

    def _generate_recommendation(self):
        # Heuristics for auto-recommendation
        hyphen_count = sum(self.stats['hyphens'].values())
        acronym_count = sum(self.stats['acronyms'].values())
        
        print("Based on stats:")
        if hyphen_count > 100:
            print("- Hyphens: HIGH FREQUENCY. Recommendation: Keep hyphenated words intact (e.g., 'supply-side' -> 'supply-side'). Splitting them might lose specific meaning.")
        else:
            print("- Hyphens: Low frequency. Recommendation: Split freely.")

        if acronym_count > 50:
            print("- Acronyms: DETECTED. Recommendation: Normalize by removing dots (U.S. -> US) to match user queries like 'US'.")


# In[4]:


# Necessary libraries
import pandas as pd
import numpy as np
import re
import csv
import math
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
import os

# Download necessary NLTK data (run once)
nltk.download('stopwords', quiet=True)

# CONFIGURATION & LOGGING
class IRConfig:
    """Central configuration for the IR System."""
    DATA_PATH = r"C:\Users\user\Documents\Sem 3\HW3\dataset\Articles.csv"
    STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    
    # BM25 Hyperparameters (Standard defaults)
    K1 = 1.5
    B = 0.75

def log(msg):
    print(f"[System Log]: {msg}")

# DATA INGESTION
def load_data(filepath):
    """
    Reads the CSV with robust handling for multi-line fields and encoding errors.
    """
    log(f"Loading dataset from {filepath}...")
    documents = []
    
    # List of encodings to try. 
    encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'ISO-8859-1']
    
    df = None
    
    for encoding in encodings_to_try:
        try:
            log(f"Attempting to load with encoding: {encoding}...")
            df = pd.read_csv(filepath, encoding=encoding)
            log(f"Success with encoding: {encoding}")
            break # Stop if successful
        except UnicodeDecodeError:
            log(f"Failed with encoding: {encoding}, trying next...")
        except Exception as e:
            log(f"Unexpected error with {encoding}: {e}")
            break

    if df is None:
        log("Critical Error: Could not read file with any supported encoding.")
        return []

    # Process the dataframe
    try:
        # Handling NaN values by replacing them with empty strings
        df = df.fillna('')
        
        for index, row in df.iterrows():
            # Robust cleaning: remove potential non-breaking spaces or weird whitespace
            clean_content = str(row['Article']).strip()
            clean_heading = str(row['Heading']).strip()
            
            doc = {
                'id': index,
                'content': clean_content,
                'heading': clean_heading,
                'date': str(row['Date']),
                'type': str(row['NewsType'])
            }
            documents.append(doc)
            
        log(f"Successfully loaded {len(documents)} documents.")
        return documents
        
    except Exception as e:
        log(f"Error parsing dataframe content: {e}")
        return []

# PREPROCESSING
def preprocess(text):
    """
    Pipeline: Lowercase -> Remove Special Chars -> Tokenize -> Remove Stopwords -> Stem
    """
    # 1. Lowercase and remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    
    # 2. Tokenize (split by whitespace)
    tokens = text.split()
    
    # 3. Stopword removal & Stemming
    clean_tokens = [
        IRConfig.STEMMER.stem(t) 
        for t in tokens 
        if t not in IRConfig.STOPWORDS
    ]
    
    return clean_tokens


# In[5]:


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(dict) # term -> {doc_id: freq}
        self.doc_lengths = {} # doc_id -> length (needed for BM25)
        self.avg_doc_length = 0
        self.total_docs = 0
        self.corpus_stats = {} # store idf later
        
    def build_index(self, documents):
        """
        Constructs the inverted index from the document corpus.
        """
        log("Building Inverted Index...")
        total_length = 0
        
        for doc in documents:
            doc_id = doc['id']
            # We index both Heading and Article Content for better recall
            full_text = f"{doc['heading']} {doc['content']}"
            tokens = preprocess(full_text)
            
            # 1. Update Document Lengths (for BM25 normalization)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # 2. Build Inverted Index
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
                
        self.total_docs = len(documents)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        log(f"Indexing complete. Vocabulary size: {len(self.inverted_index)} terms.")
        
    def get_postings(self, term):
        """Returns {doc_id: freq} for a given term."""
        return self.inverted_index.get(term, {})


# In[6]:


class RetrievalSystem:
    def __init__(self, indexer):
        self.indexer = indexer
        
    def _calculate_idf(self, term):
        """
        Calculates Inverse Document Frequency (IDF) for a term.
        Using standard log formulation.
        """
        doc_freq = len(self.indexer.get_postings(term))
        if doc_freq == 0:
            return 0
        # Standard IDF formula: log( (N - n + 0.5) / (n + 0.5) + 1 ) 
        # Adding 1 to avoid negative values
        N = self.indexer.total_docs
        return math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))

    def boolean_retrieve(self, query):
        """
        Basic AND retrieval. Returns documents containing ALL query terms.
        """
        query_terms = preprocess(query)
        if not query_terms:
            return []
        
        # Start with the set of docs for the first term
        first_term_docs = set(self.indexer.get_postings(query_terms[0]).keys())
        
        # Intersect with all other terms
        for term in query_terms[1:]:
            term_docs = set(self.indexer.get_postings(term).keys())
            first_term_docs = first_term_docs.intersection(term_docs)
            
        return list(first_term_docs)

    def bm25_rank(self, query, top_k=5):
        """
        Performs Ranked Retrieval using Okapi BM25.
        """
        query_terms = preprocess(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            postings = self.indexer.get_postings(term)
            idf = self._calculate_idf(term)
            
            for doc_id, freq in postings.items():
                # BM25 Component Calculation
                doc_len = self.indexer.doc_lengths[doc_id]
                avg_len = self.indexer.avg_doc_length
                k1 = IRConfig.K1
                b = IRConfig.B
                
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avg_len))
                
                # Accumulate score for this doc
                scores[doc_id] += idf * (numerator / denominator)
                
        # Sort by score descending
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]

    def expand_query_with_feedback(self, original_query, relevant_doc_id, alpha=1.0, beta=0.75):
            """
            Simple Rocchio-like Feedback:
            New Query Vector = Alpha * Old Query + Beta * Relevant Doc Vector
            
            In practice (for text), we append top terms from the relevant doc to the query.
            """
            # Get terms from the relevant document
            relevant_doc_terms = []
            for term, postings in self.indexer.inverted_index.items():
                if relevant_doc_id in postings:
                    relevant_doc_terms.append(term)
            
            # Simple implementation: Add top 3 most frequent terms from the doc 
            # that aren't already in the query
            current_terms = set(preprocess(original_query))
            
            # Sort doc terms by frequency in that doc
            sorted_terms = sorted(relevant_doc_terms, 
                                  key=lambda t: self.indexer.inverted_index[t][relevant_doc_id], 
                                  reverse=True)
            
            added = 0
            new_query_parts = [original_query]
            
            for term in sorted_terms:
                if term not in current_terms and added < 3:
                    new_query_parts.append(term)
                    added += 1
                    
            return " ".join(new_query_parts)


# In[7]:


# MAIN EXECUTION PIPELINE

def main():
    # 1. Initialize System
    log("Initializing CS516 IR System...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
    ROOT_DIR = os.path.dirname(BASE_DIR)                      # project root
    DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Articles.csv")
    
    df = pd.read_csv(DATA_PATH)

    filepath = r"C:\Users\user\Documents\Sem 3\HW3\dataset\Articles.csv"
    
    docs = load_data(filepath)
    if not docs:
        return

    # ... after loading docs ...
    analyzer = CorpusAnalyzer(docs)
    analyzer.analyze()
    analyzer.print_report()
    
    # Pause to let you read the report
    input("\nPress Enter to continue to Indexing based on these insights...")
    
    # 2. Build Index
    indexer = Indexer()
    indexer.build_index(docs)
    
    # 3. Init Retrieval Engine
    engine = RetrievalSystem(indexer)
    
    # 4. Interactive Loop
    while True:
        print("\n" + "="*50)
        user_query = input("Enter Query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
            
        # -- VARIANT A: Boolean Search --
        print(f"\n[Boolean Retrieval] for '{user_query}':")
        bool_results = engine.boolean_retrieve(user_query)
        if bool_results:
            for doc_id in bool_results:
                print(f" - Found in Doc ID {doc_id}: {docs[doc_id]['heading'][:50]}...")
        else:
            print(" - No exact matches found (Boolean).")

        # -- VARIANT B: Ranked Retrieval (BM25) --
        print(f"\n[Ranked Retrieval - BM25] for '{user_query}':")
        ranked_results = engine.bm25_rank(user_query)
        
        if not ranked_results:
            print(" - No relevant documents found.")
            continue
            
        for rank, (doc_id, score) in enumerate(ranked_results, 1):
            print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")
            
        # -- VARIANT C: Relevance Feedback Demo --
        # Let's assume the user liked the first result and wants "more like this"
        if ranked_results:
            top_doc_id = ranked_results[0][0]
            print(f"\n[Relevance Feedback] Assuming you liked Doc {top_doc_id}...")
            expanded_query = engine.expand_query_with_feedback(user_query, top_doc_id)
            print(f" - Optimized Query: '{expanded_query}'")
            print(" - Re-running search with optimized query...")
            
            new_results = engine.bm25_rank(expanded_query)
            for rank, (doc_id, score) in enumerate(new_results, 1):
                print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")


# In[8]:


if __name__ == "__main__":
    main()


# In[ ]:





# # Iteration 2 Lemmatizer and Corpus Analyzer based Preprocessing

# In[9]:


from nltk.stem import WordNetLemmatizer

# Download the necessary lexical database (Run once)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) # Open Multilingual Wordnet (needed for newer NLTK)

class IRConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
    ROOT_DIR = os.path.dirname(BASE_DIR)                      # project root
    DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Articles.csv")
    
    df = pd.read_csv(DATA_PATH)
    STOPWORDS = set(stopwords.words('english'))
    # CHANGE: Swapping Stemmer for Lemmatizer
    LEMMATIZER = WordNetLemmatizer() 
    
    # BM25 Hyperparameters
    K1 = 1.5
    B = 0.75


# In[10]:


def advanced_preprocess(text):
    """
    Expert Tokenizer: 
    - Keeps Hyphens (left-arm)
    - Preserves Uppercase Acronyms (US, KTI)
    - Uses Lemmatization instead of Stemming for cleaner vocabulary.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Handle Colons & Headers
    text = text.replace(':', ' ')

    # 2. Handle Acronyms (U.S. -> US)
    text = re.sub(r'\b(?:[A-Z]\.)+[A-Z]?\b', lambda m: m.group(0).replace('.', ''), text)

    # 3. Handle Possessives
    text = re.sub(r"\'s\b", "", text) 
    text = text.replace("'", "")      

    # 4. Tokenize (Split by whitespace)
    raw_tokens = text.split()

    # 5. Processing Loop
    custom_stops = {'said', 'reported', 'added', 'sources', 'also'} 
    clean_tokens = []

    for t in raw_tokens:
        # A. Clean non-alphanumeric (BUT keep hyphens)
        t_clean = re.sub(r'[^a-zA-Z0-9\-]', '', t)
        
        if not t_clean: 
            continue
            
        # B. Check for Acronym (All Caps & length > 1)
        is_acronym = t_clean.isupper() and len(t_clean) > 1
        
        # C. Case Normalization logic
        if is_acronym:
            token_to_process = t_clean # Keep "US"
        else:
            token_to_process = t_clean.lower() # "Karachi" -> "karachi"

        # D. Stopword Check
        # Convert to lower just for the check (so "The" matches "the")
        if token_to_process.lower() not in IRConfig.STOPWORDS and token_to_process.lower() not in custom_stops:
            
            # E. Lemmatization (The Big Change)
            if is_acronym:
                # Don't touch acronyms. 
                # WordNet might try to lemmatize 'US' -> 'u' if we aren't careful.
                clean_tokens.append(token_to_process)
            else:
                # Lemmatize normally. 
                # Note: Default lemmatize() assumes Noun. 
                # For an extra boost, we try Verb if it ends in 'ing' or 'ed'
                lemma = token_to_process
                if token_to_process.endswith('ing') or token_to_process.endswith('ed'):
                     lemma = IRConfig.LEMMATIZER.lemmatize(token_to_process, pos='v')
                else:
                     lemma = IRConfig.LEMMATIZER.lemmatize(token_to_process, pos='n')
                
                clean_tokens.append(lemma)

    return clean_tokens


# In[16]:


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(dict) # term -> {doc_id: freq}
        self.doc_lengths = {} # doc_id -> length (needed for BM25)
        self.avg_doc_length = 0
        self.total_docs = 0
        self.corpus_stats = {} # store idf later
        
    def build_index(self, documents):
        """
        Constructs the inverted index from the document corpus.
        """
        log("Building Inverted Index...")
        total_length = 0
        
        for doc in documents:
            doc_id = doc['id']
            # We index both Heading and Article Content for better recall
            full_text = f"{doc['heading']} {doc['content']}"
            tokens = advanced_preprocess(full_text)
            
            # 1. Update Document Lengths (for BM25 normalization)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # 2. Build Inverted Index
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
                
        self.total_docs = len(documents)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        log(f"Indexing complete. Vocabulary size: {len(self.inverted_index)} terms.")
        
    def get_postings(self, term):
        """Returns {doc_id: freq} for a given term."""
        return self.inverted_index.get(term, {})


# In[17]:


class RetrievalSystem2:
    def __init__(self, indexer):
        self.indexer = indexer
        
    def _calculate_idf(self, term):
        """
        Calculates Inverse Document Frequency (IDF) for a term.
        Using standard log formulation.
        """
        doc_freq = len(self.indexer.get_postings(term))
        if doc_freq == 0:
            return 0
        # Standard IDF formula: log( (N - n + 0.5) / (n + 0.5) + 1 ) 
        # Adding 1 to avoid negative values
        N = self.indexer.total_docs
        return math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))

    def boolean_retrieve(self, query):
        """
        Basic AND retrieval. Returns documents containing ALL query terms.
        """
        query_terms = advanced_preprocess(query)
        if not query_terms:
            return []
        
        # Start with the set of docs for the first term
        first_term_docs = set(self.indexer.get_postings(query_terms[0]).keys())
        
        # Intersect with all other terms
        for term in query_terms[1:]:
            term_docs = set(self.indexer.get_postings(term).keys())
            first_term_docs = first_term_docs.intersection(term_docs)
            
        return list(first_term_docs)

    def bm25_rank(self, query, top_k=5):
        """
        Performs Ranked Retrieval using Okapi BM25.
        """
        query_terms = advanced_preprocess(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            postings = self.indexer.get_postings(term)
            idf = self._calculate_idf(term)
            
            for doc_id, freq in postings.items():
                # BM25 Component Calculation
                doc_len = self.indexer.doc_lengths[doc_id]
                avg_len = self.indexer.avg_doc_length
                k1 = IRConfig.K1
                b = IRConfig.B
                
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avg_len))
                
                # Accumulate score for this doc
                scores[doc_id] += idf * (numerator / denominator)
                
        # Sort by score descending
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]

    def expand_query_with_feedback(self, original_query, relevant_doc_id, alpha=1.0, beta=0.75):
            """
            Simple Rocchio-like Feedback:
            New Query Vector = Alpha * Old Query + Beta * Relevant Doc Vector
            
            In practice (for text), we append top terms from the relevant doc to the query.
            """
            # Get terms from the relevant document
            relevant_doc_terms = []
            for term, postings in self.indexer.inverted_index.items():
                if relevant_doc_id in postings:
                    relevant_doc_terms.append(term)
            
            # Simple implementation: Add top 3 most frequent terms from the doc 
            # that aren't already in the query
            current_terms = set(advanced_preprocess(original_query))
            
            # Sort doc terms by frequency in that doc
            sorted_terms = sorted(relevant_doc_terms, 
                                  key=lambda t: self.indexer.inverted_index[t][relevant_doc_id], 
                                  reverse=True)
            
            added = 0
            new_query_parts = [original_query]
            
            for term in sorted_terms:
                if term not in current_terms and added < 3:
                    new_query_parts.append(term)
                    added += 1
                    
            return " ".join(new_query_parts)


# In[18]:


# MAIN EXECUTION PIPELINE

def main():
    # 1. Initialize System
    log("Initializing CS516 IR System...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
    ROOT_DIR = os.path.dirname(BASE_DIR)                      # project root
    DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Articles.csv")
    
    docs = load_data(filepath)
    if not docs:
        return

    # ... after loading docs ...
    analyzer = CorpusAnalyzer(docs)
    analyzer.analyze()
    analyzer.print_report()
    
    # Pause to let you read the report
    input("\nPress Enter to continue to Indexing based on these insights...")
    
    # 2. Build Index
    indexer = Indexer()
    indexer.build_index(docs)
    
    # 3. Init Retrieval Engine
    engine = RetrievalSystem2(indexer)
    
    # 4. Interactive Loop
    while True:
        print("\n" + "="*50)
        user_query = input("Enter Query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
            
        # Boolean Search
        print(f"\n[Boolean Retrieval] for '{user_query}':")
        bool_results = engine.boolean_retrieve(user_query)
        if bool_results:
            for doc_id in bool_results:
                print(f" - Found in Doc ID {doc_id}: {docs[doc_id]['heading'][:50]}...")
        else:
            print(" - No exact matches found (Boolean).")

        # Ranked Retrieval (BM25)
        print(f"\n[Ranked Retrieval - BM25] for '{user_query}':")
        ranked_results = engine.bm25_rank(user_query)
        
        if not ranked_results:
            print(" - No relevant documents found.")
            continue
            
        for rank, (doc_id, score) in enumerate(ranked_results, 1):
            print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")
            
        # Relevance Feedback
        # Let's assume the user liked the first result and wants "more like this"
        if ranked_results:
            top_doc_id = ranked_results[0][0]
            print(f"\n[Relevance Feedback] Assuming you liked Doc {top_doc_id}...")
            expanded_query = engine.expand_query_with_feedback(user_query, top_doc_id)
            print(f" - Optimized Query: '{expanded_query}'")
            print(" - Re-running search with optimized query...")
            
            new_results = engine.bm25_rank(expanded_query)
            for rank, (doc_id, score) in enumerate(new_results, 1):
                print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")


# In[19]:


if __name__ == "__main__":
    main()


# In[ ]:





# #Iteration-3 Domain specific stop words

# In[23]:


import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
ROOT_DIR = os.path.dirname(BASE_DIR)                      # project root
file_path = os.path.join(ROOT_DIR, "dataset", "Articles.csv")
docs = load_data(filepath)

class CorpusAnalyzer2:
    def __init__(self, documents):
        self.documents = documents
        self.stats = {
            "hyphens": Counter(),
            "apostrophes": Counter(),
            "acronyms": Counter(),
            "colons": Counter(),
            "numbers": Counter()
        }

    def analyze(self):
        print("Analyzing corpus for special token patterns...")
        
        for doc in self.documents:
            text = f"{doc['heading']} {doc['content']}"
            
            # 1. Hyphenated words (e.g., "supply-side", "co-operation")
            # Matches word-word but excludes simple minuses between numbers
            hyphens = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b', text)
            self.stats["hyphens"].update(hyphens)
            
            # 2. Apostrophes (Possessives vs Contractions)
            # Matches word's or word't, etc.
            apostrophes = re.findall(r'\b[a-zA-Z]+\'[a-zA-Z]+\b', text)
            self.stats["apostrophes"].update(apostrophes)
            
            # 3. Acronyms / Abbreviations with dots (e.g., U.S., U.N., K.T.I.)
            # Matches capital letter followed by dot, repeated
            acronyms = re.findall(r'\b(?:[A-Z]\.)+[A-Z]?\b', text)
            self.stats["acronyms"].update(acronyms)
            
            # 4. Colons (often used in news for "KARACHI: ...")
            # Matches Word: at the start of lines or sentences
            colons = re.findall(r'\b[A-Z][a-zA-Z]*:', text)
            self.stats["colons"].update(colons)

    def print_report(self):
        print("\n=== CORPUS ANALYSIS REPORT ===")
        
        print(f"\n[Hyphenated Words] Top 10 of {len(self.stats['hyphens'])} unique:")
        for w, c in self.stats['hyphens'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Apostrophes] Top 10 of {len(self.stats['apostrophes'])} unique:")
        for w, c in self.stats['apostrophes'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Acronyms] Top 10 of {len(self.stats['acronyms'])} unique:")
        for w, c in self.stats['acronyms'].most_common(10):
            print(f"  {w}: {c}")
            
        print(f"\n[Colons/Headers] Top 10:")
        for w, c in self.stats['colons'].most_common(10):
            print(f"  {w}: {c}")

        print("\n=== RECOMMENDATION ===")
        self._generate_recommendation()

    def _generate_recommendation(self):
        # Heuristics for auto-recommendation
        hyphen_count = sum(self.stats['hyphens'].values())
        acronym_count = sum(self.stats['acronyms'].values())
        
        print("Based on stats:")
        if hyphen_count > 100:
            print("- Hyphens: HIGH FREQUENCY. Recommendation: Keep hyphenated words intact (e.g., 'supply-side' -> 'supply-side'). Splitting them might lose specific meaning.")
        else:
            print("- Hyphens: Low frequency. Recommendation: Split freely.")

        if acronym_count > 50:
            print("- Acronyms: DETECTED. Recommendation: Normalize by removing dots (U.S. -> US) to match user queries like 'US'.")

    def analyze_top_frequent_terms(documents, top_n=50):
        """
        Scans the corpus to find the most frequent terms.
        Helps in identifying domain-specific stopwords.
        """
        print(f"\n[Analysis]: Scanning corpus for top {top_n} frequent terms...")
        
        # We use a Counter to track term frequency across the entire collection
        corpus_freq = Counter()
        
        # We want to check frequencies AFTER standard stopwords are removed 
        # but BEFORE your custom list is applied, to see what 'leaks' through.
        
        # Temporary set of standard NLTK stopwords for filtering
        standard_stops = IRConfig.STOPWORDS 
        
        for doc in documents:
            # Use a simplified tokenizer here to just get raw words
            # We simulate the process: Lowercase -> Split -> Remove Standard Stops
            text = f"{doc['heading']} {doc['content']}".lower()
            # Remove simple punctuation for accurate counting
            text = re.sub(r'[^a-z0-9\s]', '', text)
            tokens = text.split()
            
            # Filter only standard English stopwords to see what remains
            filtered = [t for t in tokens if t not in standard_stops]
            corpus_freq.update(filtered)
            
        print(f"\n--- TOP {top_n} MOST FREQUENT TERMS (Candidates for Custom Stopwords) ---")
        print(f"{'Term':<20} | {'Frequency':<10} | {' Recommendation'}")
        print("-" * 50)
        
        for term, freq in corpus_freq.most_common(top_n):
            # Heuristic: If it looks like a reporting verb or generic noun, we flag it
            recommendation = ""
            common_news_stops = {'said', 'reported', 'added', 'also', 'sources', 'share', 'percent', 'year', 'new', 'two'}
            
            if term in common_news_stops:
                recommendation = "[ALREADY CAUGHT]"
            elif freq > len(documents) * 0.1: # If term appears in >10% of docs (rough heuristic)
                recommendation = "-> CONSIDER ADDING"
                
            print(f"{term:<20} | {freq:<10} | {recommendation}")
    
    # --- RUN THIS INTERACTIVELY ---
    analyze_top_frequent_terms(docs)


# In[28]:


# --- MAIN EXECUTION PIPELINE ---

def main():
    # 1. Initialize System
    log("Initializing CS516 IR System...")
    
    # NOTE: You would use the path provided in your prompt:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
    ROOT_DIR = os.path.dirname(BASE_DIR)                      # project root
    file_path = os.path.join(ROOT_DIR, "dataset", "Articles.csv")
    
    docs = load_data(filepath)
    if not docs:
        return

    # ... after loading docs ...
    analyzer = CorpusAnalyzer2(docs)
    analyzer.analyze()
    analyzer.print_report()
    
    # Pause to let you read the report
    input("\nPress Enter to continue to Indexing based on these insights...")
    
    # 2. Build Index
    indexer = Indexer()
    indexer.build_index(docs)
    
    # 3. Init Retrieval Engine
    engine = RetrievalSystem2(indexer)
    
    # 4. Interactive Loop
    while True:
        print("\n" + "="*50)
        user_query = input("Enter Query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
            
        # -- VARIANT A: Boolean Search --
        print(f"\n[Boolean Retrieval] for '{user_query}':")
        bool_results = engine.boolean_retrieve(user_query)
        if bool_results:
            for doc_id in bool_results:
                print(f" - Found in Doc ID {doc_id}: {docs[doc_id]['heading'][:50]}...")
        else:
            print(" - No exact matches found (Boolean).")

        # -- VARIANT B: Ranked Retrieval (BM25) --
        print(f"\n[Ranked Retrieval - BM25] for '{user_query}':")
        ranked_results = engine.bm25_rank(user_query)
        
        if not ranked_results:
            print(" - No relevant documents found.")
            continue
            
        for rank, (doc_id, score) in enumerate(ranked_results, 1):
            print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")
            
        # -- VARIANT C: Relevance Feedback Demo --
        # Let's assume the user liked the first result and wants "more like this"
        if ranked_results:
            top_doc_id = ranked_results[0][0]
            print(f"\n[Relevance Feedback] Assuming you liked Doc {top_doc_id}...")
            expanded_query = engine.expand_query_with_feedback(user_query, top_doc_id)
            print(f" - Optimized Query: '{expanded_query}'")
            print(" - Re-running search with optimized query...")
            
            new_results = engine.bm25_rank(expanded_query)
            for rank, (doc_id, score) in enumerate(new_results, 1):
                print(f" {rank}. Doc {doc_id} (Score: {score:.4f}) | {docs[doc_id]['heading']}")


# In[29]:


if __name__ == "__main__":
    main()


# In[26]:


import random

class SimpleEvaluator:
    def __init__(self, retrieval_system, documents):
        self.system = retrieval_system
        self.documents = documents

    def run_known_item_test(self, num_queries=20, top_k=5):
        """
        Picks 'num_queries' random documents.
        Uses their 'heading' as the query.
        Checks if the correct document ID appears in the top 'top_k' results.
        Returns: Recall@K Score (0.0 to 1.0)
        """
        print(f"\n--- RUNNING KNOWN-ITEM EVALUATION (n={num_queries}, k={top_k}) ---")
        
        hits = 0
        
        # Select random documents to test
        test_docs = random.sample(self.documents, min(num_queries, len(self.documents)))
        
        for i, doc in enumerate(test_docs):
            target_id = doc['id']
            query = doc['heading']
            
            # Run the search
            results = self.system.bm25_rank(query, top_k=top_k)
            retrieved_ids = [r[0] for r in results]
            
            # Check if our target document is in the retrieved list
            if target_id in retrieved_ids:
                hits += 1
                status = "HIT"
            else:
                status = "MISS"
                
            # Print first 5 for sanity check (optional)
            if i < 5: 
                print(f"Query: '{query[:30]}...' -> {status}")

        recall_score = hits / num_queries
        print("-" * 40)
        print(f"Success Rate (Recall@{top_k}): {recall_score:.2%} ({hits}/{num_queries})")
        print("-" * 40)
        return recall_score

# --- HOW TO RUN IT ---
# Add this inside your main() loop or run separately:

evaluator = SimpleEvaluator(engine, docs)
score = evaluator.run_known_item_test()


# In[ ]:




