import json
import numpy as np

import json
import logging
import numpy as np
import re
import os
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="debug.log")
logger = logging.getLogger(__name__)

# Globals
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

def load_corpus(corpus_file):
    try:
        with open(corpus_file, "r") as file:
            corpus_dict = json.load(file)
        logger.info(f"Corpus loaded from {corpus_file}.")
        return corpus_dict
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return {}

def get_cache_paths(corpus_file):
    base = os.path.splitext(os.path.basename(corpus_file))[0]
    vectorizer_path = f"cache/{base}_vectorizer.pkl"
    matrix_path = f"cache/{base}_matrix.pkl"
    keyword_path = f"cache/{base}_keywords.pkl"
    return vectorizer_path, matrix_path, keyword_path

def generate_esa_vectors(text, corpus_file):
    """
    Generate ESA vectors for the given text using the lemmatized corpus.
    Args:
        text (str): The input text to generate ESA vectors for.
        corpus_file (str): Path to the corpus file.
    Returns:
        list: The ESA vectors for the input text.
    """
    logger.info("Generating ESA vectors.")
    
    corpus = load_corpus(corpus_file)  # Load the corpus
    if not corpus:
        logger.error("Corpus is empty or could not be loaded.")
        return [], []

    # Preprocess text and corpus
    sentences = sent_tokenize(text)
    processed_sentences = [preprocess_text(s) for s in sentences]
    processed_corpus = list(corpus.values())
    all_documents = processed_sentences + processed_corpus

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Generate ESA vectors for each processed sentence
    esa_vectors = []
    for i in range(len(processed_sentences)):
        similarities = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[len(processed_sentences):])
        esa_vector = similarities.flatten()
        esa_vectors.append(esa_vector)
    
    if esa_vectors:
        mean_vector = np.mean(esa_vectors, axis=0)
        return mean_vector.tolist()
    else:
        logger.error("No ESA vectors generated.")
        return []



# def main():
#     text = "What are the structural and aeroelastic problems associated with flight of high speed aircraft?"
#     esa_vector, esa_dict = generate_esa_vectors(text, corpus_file="topics/topics_top_100_lem_wiki.json")

#     if esa_vector:
#         print("ESA Vector (first 10 values):", esa_vector[:10])
#         print("\nTop 5 Concepts:")
#         for concept, score in sorted(esa_dict.items(), key=lambda x: -x[1])[:5]:
#             print(f"{concept}: {score:.4f}")
#     else:
#         print("ESA vector generation failed.")

# if __name__ == "__main__":
#     main()


class ESAInformationRetrieval:
    
    def __init__(self, corpus_size=1000):
        
        self.corpus_size = corpus_size
        self.docIDs = []
        self.docVectors = None
    
    def build_vectors(self):
        
        with open(f"topics/topics_top_{self.corpus_size}_docs_esa.json", "r") as f:
            doc_esa_vectors = json.load(f)
            f.close()
        self.docIDs = [int(doc_id) for doc_id in doc_esa_vectors.keys()]
        self.docVectors = np.array([x for x in list(doc_esa_vectors.values()) if len(x) != 0])
    
    def rank(self, query):
        
        queryVector = generate_esa_vectors(query, corpus_file=f"topics/topics_top_{self.corpus_size}_lem_wiki.json")
                
        sim_matrix = cosine_similarity(np.array(queryVector).reshape(1, -1), self.docVectors)
              
        for row in sim_matrix:
            ranked_indices = np.argsort(-row)
            ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
            
        return ranked_docIDs