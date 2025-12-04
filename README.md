# IR-system-v1-msds24039
Implementation of an Information Retrieval (IR) System for CS516 (Fall 2025), including preprocessing pipeline evolution, tokenization strategies, vector space modeling, ranking, and system architecture.
CS516 – Information Retrieval System (Fall 2025)
Information Technology University — Dr. Ahmad Mustafa
📌 Overview

This repository contains the full implementation, architecture, and documentation of an Information Retrieval (IR) system developed for CS516: Information Retrieval & Text Mining.
The system is built around a real-world dataset of textual articles and implements:

Robust text preprocessing

Evolving tokenization pipeline (3 iterations)

Inverted index construction

TF–IDF weighting

Vector Space Retrieval Model

Cosine similarity ranking

A modular and extensible IR system architecture.

The project is designed for clarity, reproducibility, and research-style experimentation.


🧠 Conceptual Summary

This IR system is built using the Vector Space Model (VSM). Each document is converted into a weighted vector using TF-IDF, and user queries are processed through the same pipeline.
Similarity between query and documents is computed using cosine similarity, and results are ranked accordingly.

🔍 Three-Iteration Tokenization Evolution

The preprocessing evolved across three experimental iterations, each improving vocabulary quality and retrieval effectiveness.

⭐ Iteration 1 — Basic Cleaning & Tokenization

Operations performed:

Lowercasing

Removing punctuation

Splitting on spaces

Removing stopwords

Basic token list output

Outcome:

Vocabulary retained too many noisy forms: plurals, verb variations, domain-irrelevant terms.

High sparsity in document vectors.

Retrieval inconsistent for mixed morphological forms.

⭐ Iteration 2 — Improved Normalization Pipeline

Enhancements:

Lemmatization (WordNetLemmatizer)

Normalized whitespace handling

Improved stopword list + domain-specific additions

Digit removal

Removal of tokens <3 characters

Outcome:

Vocabulary reduced significantly (~18–25%).

Higher term overlap across documents → better cosine similarity scores.

Reduced noise, but still contained domain synonyms treated independently.

⭐ Iteration 3 — Final Tokenization Pipeline (Production Level)

Introduced the strongest refinements:

🔧 Pipeline Steps

Lowercasing

Regex-based punctuation & symbol removal

Tokenization using spaCy tokenizer (robust handling)

Lemmatization using spaCy linguistic models

POS-aware lemmatization (v → base verb, n → singular noun, etc.)

Domain-specific stopwords:

article, news, report, said, will, said_that, etc.

Removal of:

numbers

single-character tokens

rare words (<2 document occurrences)

Optional bigram/trigram formation for domain terms

Outcome:

Clean, compact, semantically stronger vocabulary.

~40–55% vocabulary reduction vs. raw text.

More stable and accurate ranking behavior.

Better query–document matching due to consistent lemmas.
