# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
# import nltk
# from nltk.corpus import stopwords
# from nltk import pos_tag
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from evaluation import Evaluation

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# class LSASearchEngine:
#     """
#     Implements Latent Semantic Analysis for information retrieval
#     """
    
#     def __init__(self, n_components=300):
#         """
#         Initialize the LSA search engine
        
#         Parameters:
#         -----------
#         n_components : int
#             Number of components for SVD
#         """
#         self.n_components = n_components
#         self.vectorizer = TfidfVectorizer()
#         self.svd = TruncatedSVD(n_components=n_components, random_state=42)
#         self.docs_svd = None
#         self.doc_ids = None
        
#     def preprocess_text(self, text, remove_stopwords=True, pos_tag_text=False):
#         """
#         Preprocess text by tokenizing, removing stopwords (optional),
#         and adding POS tags (optional)
        
#         Parameters:
#         -----------
#         text : str
#             Text to be preprocessed
#         remove_stopwords : bool
#             Whether to remove stopwords
#         pos_tag_text : bool
#             Whether to add POS tags
            
#         Returns:
#         --------
#         str
#             Preprocessed text
#         """
#         tokens = word_tokenize(text.lower())
        
#         if remove_stopwords:
#             stop_words = set(stopwords.words('english'))
#             tokens = [token for token in tokens if token not in stop_words]
        
#         if pos_tag_text:
#             pos_tags = pos_tag(tokens)
#             return " ".join([f"{word}_{tag}" for word, tag in pos_tags])
#         else:
#             return " ".join(tokens)
    
#     def build_index(self, docs, doc_ids, use_pos_tagging=False):
#         """
#         Build the LSA index from documents
        
#         Parameters:
#         -----------
#         docs : list
#             List of document texts
#         doc_ids : list
#             List of document IDs
#         use_pos_tagging : bool
#             Whether to use POS tagging for documents
            
#         Returns:
#         --------
#         None
#         """
#         self.doc_ids = doc_ids
        
#         # Preprocess docs (with or without POS tagging)
#         if use_pos_tagging:
#             preprocessed_docs = [self.preprocess_text(doc, pos_tag_text=True) for doc in docs]
#         else:
#             preprocessed_docs = [self.preprocess_text(doc) for doc in docs]
        
#         # Build TF-IDF matrix
#         self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_docs)
        
#         # Apply SVD for LSA
#         self.docs_svd = self.svd.fit_transform(self.tfidf_matrix)
        
#         print(f"LSA index built with {self.n_components} components")
#         print(f"Explained variance ratio: {sum(self.svd.explained_variance_ratio_):.2f}")
        
#     def search(self, queries, use_pos_tagging=False):
#         """
#         Search using LSA
        
#         Parameters:
#         -----------
#         queries : list
#             List of queries
#         use_pos_tagging : bool
#             Whether to use POS tagging for queries
            
#         Returns:
#         --------
#         list
#             List of ordered document IDs for each query
#         """
#         # Preprocess queries (with or without POS tagging)
#         if use_pos_tagging:
#             preprocessed_queries = [self.preprocess_text(query, pos_tag_text=True) for query in queries]
#         else:
#             preprocessed_queries = [self.preprocess_text(query) for query in queries]
        
#         # Transform queries to TF-IDF
#         queries_tfidf = self.vectorizer.transform(preprocessed_queries)
        
#         # Project into LSA space
#         queries_svd = self.svd.transform(queries_tfidf)
        
#         # Calculate cosine similarity
#         cosine_similarities = []
#         for query_vector in queries_svd:
#             # Dot product / (norm(a) * norm(b))
#             doc_norms = np.linalg.norm(self.docs_svd, axis=1)
#             query_norm = np.linalg.norm(query_vector)
            
#             similarities = np.zeros(len(self.docs_svd))
#             for i, doc_vector in enumerate(self.docs_svd):
#                 if doc_norms[i] * query_norm != 0:  # Avoid division by zero
#                     similarities[i] = np.dot(doc_vector, query_vector) / (doc_norms[i] * query_norm)
            
#             cosine_similarities.append(similarities)
        
#         # Sort documents by similarity for each query
#         doc_IDs_ordered = []
#         for similarities in cosine_similarities:
#             sorted_indices = np.argsort(-similarities)
#             ordered_docs = [self.doc_ids[idx] for idx in sorted_indices]
#             doc_IDs_ordered.append(ordered_docs)
        
#         return doc_IDs_ordered
    
#     def find_optimal_components(self, docs, doc_ids, queries, query_ids, qrels, 
#                               min_components=100, max_components=1500, step=100):
#         """
#         Find the optimal number of components for LSA
        
#         Parameters:
#         -----------
#         docs : list
#             List of document texts
#         doc_ids : list
#             List of document IDs
#         queries : list
#             List of queries
#         query_ids : list
#             List of query IDs
#         qrels : list
#             List of relevance judgments
#         min_components : int
#             Minimum number of components to try
#         max_components : int
#             Maximum number of components to try
#         step : int
#             Step size for components
            
#         Returns:
#         --------
#         int
#             Optimal number of components
#         """
#         components = list(range(min_components, max_components + 1, step))
#         eval_metrics = Evaluation()
#         maps = []
        
#         for n_comp in components:
#             print(f"Testing with {n_comp} components...")
#             self.n_components = n_comp
#             self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            
#             self.build_index(docs, doc_ids)
#             doc_IDs_ordered = self.search(queries)
            
#             # Calculate MAP@10
#             map_score = eval_metrics.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k=10)
#             maps.append(map_score)
#             print(f"MAP@10 = {map_score}")
        
#         # Plot MAP vs number of components
#         plt.figure(figsize=(10, 6))
#         plt.plot(components, maps, marker='o')
#         plt.xlabel('Number of Components')
#         plt.ylabel('MAP@10')
#         plt.title('Performance vs LSA Components')
#         plt.grid(True)
#         plt.savefig('lsa_components_performance.png')
#         plt.show()
        
#         # Return the optimal number of components
#         optimal_components = components[np.argmax(maps)]
#         print(f"Optimal number of components: {optimal_components}")
#         return optimal_components
    
#     def evaluate_performance(self, doc_IDs_ordered, query_ids, qrels):
#         """
#         Evaluate the performance of the LSA search engine
        
#         Parameters:
#         -----------
#         doc_IDs_ordered : list
#             List of ordered document IDs for each query
#         query_ids : list
#             List of query IDs
#         qrels : list
#             List of relevance judgments
            
#         Returns:
#         --------
#         dict
#             Dictionary of evaluation metrics
#         """
#         eval_metrics = Evaluation()
#         results = {}
        
#         k_values = [1, 3, 5, 10]
#         for k in k_values:
#             precision = eval_metrics.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
#             recall = eval_metrics.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
#             fscore = eval_metrics.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
#             map_score = eval_metrics.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
#             ndcg = eval_metrics.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            
#             results[f'k={k}'] = {
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F-score': fscore,
#                 'MAP': map_score,
#                 'nDCG': ndcg
#             }
            
#             print(f"Results at k={k}:")
#             print(f"  Precision: {precision:.4f}")
#             print(f"  Recall: {recall:.4f}")
#             print(f"  F-score: {fscore:.4f}")
#             print(f"  MAP: {map_score:.4f}")
#             print(f"  nDCG: {ndcg:.4f}")
        
#         # Plot metrics
#         plt.figure(figsize=(12, 8))
#         metrics = ['Precision', 'Recall', 'F-score', 'MAP', 'nDCG']
#         for metric in metrics:
#             values = [results[f'k={k}'][metric] for k in k_values]
#             plt.plot(k_values, values, marker='o', label=metric)
        
#         plt.xlabel('k')
#         plt.ylabel('Score')
#         plt.title('LSA Performance Metrics')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig('lsa_performance_metrics.png')
#         plt.show()
        
#         return results
    
#     def compare_models(self, models_results, model_names):
#         """
#         Compare different models using hypothesis testing
        
#         Parameters:
#         -----------
#         models_results : list of lists
#             List of lists of results for each model at k=10
#         model_names : list
#             List of model names
            
#         Returns:
#         --------
#         None
#         """
#         metrics = ['Precision', 'Recall', 'F-score', 'nDCG']
        
#         for i in range(len(model_names)):
#             for j in range(i+1, len(model_names)):
#                 print(f"\nComparing {model_names[i]} vs {model_names[j]}:")
                
#                 for metric in metrics:
#                     a = models_results[i][metric]
#                     b = models_results[j][metric]
                    
#                     # Perform t-test
#                     t_stat, p_value = stats.ttest_rel(a, b)
                    
#                     print(f"  {metric}: t = {t_stat:.4f}, p = {p_value:.4f}")
#                     if p_value < 0.05:
#                         better = model_names[i] if np.mean(a) > np.mean(b) else model_names[j]
#                         print(f"    Significant difference! {better} is better.")
#                     else:
#                         print("    No significant difference.")

# def main():
#     # Load data
#     print("Loading data...")
#     queries_json = json.load(open("./cran_queries.json", 'r'))[:]
#     query_ids = [item["query number"] for item in queries_json]
#     queries = [item["query"] for item in queries_json]
    
#     docs_json = json.load(open("./cran_docs.json", 'r'))[:]
#     doc_ids = [item["id"] for item in docs_json]
#     docs = [item["body"] for item in docs_json]
    
#     qrels = json.load(open("./cran_qrels.json", 'r'))[:]
    
#     # Initialize LSA engine
#     lsa_engine = LSASearchEngine(n_components=300)
    
#     # Find optimal number of components
#     print("\nFinding optimal number of components...")
#     optimal_components = lsa_engine.find_optimal_components(
#         docs, doc_ids, queries, query_ids, qrels,
#         min_components=100, max_components=1500, step=100
#     )
    
#     # Build index with optimal number of components
#     print(f"\nBuilding LSA index with {optimal_components} components...")
#     lsa_engine.n_components = optimal_components
#     lsa_engine.svd = TruncatedSVD(n_components=optimal_components, random_state=42)
#     lsa_engine.build_index(docs, doc_ids)
    
#     # Standard LSA
#     print("\nEvaluating standard LSA...")
#     doc_IDs_ordered_lsa = lsa_engine.search(queries)
#     results_lsa = lsa_engine.evaluate_performance(doc_IDs_ordered_lsa, query_ids, qrels)
    
#     # POS-tagged LSA
#     print("\nEvaluating POS-tagged LSA...")
#     lsa_engine.build_index(docs, doc_ids, use_pos_tagging=True)
#     doc_IDs_ordered_pos_lsa = lsa_engine.search(queries, use_pos_tagging=True)
#     results_pos_lsa = lsa_engine.evaluate_performance(doc_IDs_ordered_pos_lsa, query_ids, qrels)
    
#     # Extract per-query metrics for comparison
#     query_metrics_lsa = {'Precision': [], 'Recall': [], 'F-score': [], 'nDCG': []}
#     query_metrics_pos_lsa = {'Precision': [], 'Recall': [], 'F-score': [], 'nDCG': []}
    
#     eval_metrics = Evaluation()
#     k = 10  # Using k=10 for comparison
    
#     # Get per-query metrics for standard LSA
#     for i, query_id in enumerate(query_ids):
#         true_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(query_id)]
#         query_docs = doc_IDs_ordered_lsa[i][:k]
        
#         query_metrics_lsa['Precision'].append(eval_metrics.queryPrecision(query_docs, query_id, true_docs, k))
#         query_metrics_lsa['Recall'].append(eval_metrics.queryRecall(query_docs, query_id, true_docs, k))
#         query_metrics_lsa['F-score'].append(eval_metrics.queryFscore(query_docs, query_id, true_docs, k))
#         query_metrics_lsa['nDCG'].append(eval_metrics.queryNDCG(query_docs, query_id, true_docs, k))
    
#     # Get per-query metrics for POS-tagged LSA
#     for i, query_id in enumerate(query_ids):
#         true_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(query_id)]
#         query_docs = doc_IDs_ordered_pos_lsa[i][:k]
        
#         query_metrics_pos_lsa['Precision'].append(eval_metrics.queryPrecision(query_docs, query_id, true_docs, k))
#         query_metrics_pos_lsa['Recall'].append(eval_metrics.queryRecall(query_docs, query_id, true_docs, k))
#         query_metrics_pos_lsa['F-score'].append(eval_metrics.queryFscore(query_docs, query_id, true_docs, k))
#         query_metrics_pos_lsa['nDCG'].append(eval_metrics.queryNDCG(query_docs, query_id, true_docs, k))
    
#     # Compare standard LSA vs POS-tagged LSA
#     print("\nComparing Standard LSA vs POS-tagged LSA...")
#     lsa_engine.compare_models(
#         [query_metrics_lsa, query_metrics_pos_lsa],
#         ["Standard LSA", "POS-tagged LSA"]
#     )
    
#     # Save results to a file
#     results = {
#         'Standard LSA': results_lsa,
#         'POS-tagged LSA': results_pos_lsa,
#         'Optimal Components': optimal_components
#     }
    
#     with open('lsa_results.json', 'w') as f:
#         json.dump(results, f, indent=4)
    
#     print("\nAnalysis complete. Results saved to lsa_results.json")

# if __name__ == "__main__":
#     main()
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from evaluation import Evaluation

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class LSASearchEngine:
    """
    Implements Latent Semantic Analysis for information retrieval
    """
    
    def __init__(self, n_components=300):
        """
        Initialize the LSA search engine
        
        Parameters:
        -----------
        n_components : int
            Number of components for SVD
        """
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.docs_svd = None
        self.doc_ids = None
        
    def preprocess_text(self, text, remove_stopwords=True, pos_tag_text=False):
        """
        Preprocess text by tokenizing, removing stopwords (optional),
        and adding POS tags (optional)
        
        Parameters:
        -----------
        text : str
            Text to be preprocessed
        remove_stopwords : bool
            Whether to remove stopwords
        pos_tag_text : bool
            Whether to add POS tags
            
        Returns:
        --------
        str
            Preprocessed text
        """
        tokens = word_tokenize(text.lower())
        
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        if pos_tag_text:
            pos_tags = pos_tag(tokens)
            return " ".join([f"{word}_{tag}" for word, tag in pos_tags])
        else:
            return " ".join(tokens)
    
    def build_index(self, docs, doc_ids, use_pos_tagging=False):
        """
        Build the LSA index from documents
        
        Parameters:
        -----------
        docs : list
            List of document texts
        doc_ids : list
            List of document IDs
        use_pos_tagging : bool
            Whether to use POS tagging for documents
            
        Returns:
        --------
        None
        """
        self.doc_ids = doc_ids
        
        # Preprocess docs (with or without POS tagging)
        if use_pos_tagging:
            preprocessed_docs = [self.preprocess_text(doc, pos_tag_text=True) for doc in docs]
        else:
            preprocessed_docs = [self.preprocess_text(doc) for doc in docs]
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_docs)
        
        # Apply SVD for LSA
        self.docs_svd = self.svd.fit_transform(self.tfidf_matrix)
        
        print(f"LSA index built with {self.n_components} components")
        print(f"Explained variance ratio: {sum(self.svd.explained_variance_ratio_):.2f}")
        
    def search(self, queries, use_pos_tagging=False):
        """
        Search using LSA
        
        Parameters:
        -----------
        queries : list
            List of queries
        use_pos_tagging : bool
            Whether to use POS tagging for queries
            
        Returns:
        --------
        list
            List of ordered document IDs for each query
        """
        # Preprocess queries (with or without POS tagging)
        if use_pos_tagging:
            preprocessed_queries = [self.preprocess_text(query, pos_tag_text=True) for query in queries]
        else:
            preprocessed_queries = [self.preprocess_text(query) for query in queries]
        
        # Transform queries to TF-IDF
        queries_tfidf = self.vectorizer.transform(preprocessed_queries)
        
        # Project into LSA space
        queries_svd = self.svd.transform(queries_tfidf)
        
        # Calculate cosine similarity
        cosine_similarities = []
        for query_vector in queries_svd:
            # Dot product / (norm(a) * norm(b))
            doc_norms = np.linalg.norm(self.docs_svd, axis=1)
            query_norm = np.linalg.norm(query_vector)
            
            similarities = np.zeros(len(self.docs_svd))
            for i, doc_vector in enumerate(self.docs_svd):
                if doc_norms[i] * query_norm != 0:  # Avoid division by zero
                    similarities[i] = np.dot(doc_vector, query_vector) / (doc_norms[i] * query_norm)
            
            cosine_similarities.append(similarities)
        
        # Sort documents by similarity for each query
        doc_IDs_ordered = []
        for similarities in cosine_similarities:
            sorted_indices = np.argsort(-similarities)
            ordered_docs = [self.doc_ids[idx] for idx in sorted_indices]
            doc_IDs_ordered.append(ordered_docs)
        
        return doc_IDs_ordered
    
    def find_optimal_components(self, docs, doc_ids, queries, query_ids, qrels, 
                              min_components=100, max_components=1500, step=100):
        """
        Find the optimal number of components for LSA
        
        Parameters:
        -----------
        docs : list
            List of document texts
        doc_ids : list
            List of document IDs
        queries : list
            List of queries
        query_ids : list
            List of query IDs
        qrels : list
            List of relevance judgments
        min_components : int
            Minimum number of components to try
        max_components : int
            Maximum number of components to try
        step : int
            Step size for components
            
        Returns:
        --------
        int
            Optimal number of components
        """
        components = list(range(min_components, max_components + 1, step))
        eval_metrics = Evaluation()
        maps = []
        
        for n_comp in components:
            print(f"Testing with {n_comp} components...")
            self.n_components = n_comp
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            
            self.build_index(docs, doc_ids)
            doc_IDs_ordered = self.search(queries)
            
            # Calculate MAP@10
            map_score = eval_metrics.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k=10)
            maps.append(map_score)
            print(f"MAP@10 = {map_score}")
        
        # Plot MAP vs number of components
        plt.figure(figsize=(10, 6))
        plt.plot(components, maps, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('MAP@10')
        plt.title('Performance vs LSA Components')
        plt.grid(True)
        plt.savefig('lsa_components_performance.png')
        plt.show()
        
        # Return the optimal number of components
        optimal_components = components[np.argmax(maps)]
        print(f"Optimal number of components: {optimal_components}")
        return optimal_components
    
    def evaluate_performance(self, doc_IDs_ordered, query_ids, qrels):
        """
        Evaluate the performance of the LSA search engine
        
        Parameters:
        -----------
        doc_IDs_ordered : list
            List of ordered document IDs for each query
        query_ids : list
            List of query IDs
        qrels : list
            List of relevance judgments
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        eval_metrics = Evaluation()
        results = {}
        
        k_values = [1, 3, 5, 10]
        for k in k_values:
            precision = eval_metrics.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
            recall = eval_metrics.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            fscore = eval_metrics.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            map_score = eval_metrics.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
            ndcg = eval_metrics.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            
            results[f'k={k}'] = {
                'Precision': precision,
                'Recall': recall,
                'F-score': fscore,
                'MAP': map_score,
                'nDCG': ndcg
            }
            
            print(f"Results at k={k}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F-score: {fscore:.4f}")
            print(f"  MAP: {map_score:.4f}")
            print(f"  nDCG: {ndcg:.4f}")
        
        # Plot metrics
        plt.figure(figsize=(12, 8))
        metrics = ['Precision', 'Recall', 'F-score', 'MAP', 'nDCG']
        for metric in metrics:
            values = [results[f'k={k}'][metric] for k in k_values]
            plt.plot(k_values, values, marker='o', label=metric)
        
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.title('LSA Performance Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig('lsa_performance_metrics.png')
        plt.show()
        
        return results
    
    def compare_models(self, models_results, model_names):
        """
        Compare different models using hypothesis testing
        
        Parameters:
        -----------
        models_results : list of lists
            List of lists of results for each model at k=10
        model_names : list
            List of model names
            
        Returns:
        --------
        None
        """
        metrics = ['Precision', 'Recall', 'F-score', 'nDCG']
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                print(f"\nComparing {model_names[i]} vs {model_names[j]}:")
                
                for metric in metrics:
                    a = models_results[i][metric]
                    b = models_results[j][metric]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_rel(a, b)
                    
                    print(f"  {metric}: t = {t_stat:.4f}, p = {p_value:.4f}")
                    if p_value < 0.05:
                        better = model_names[i] if np.mean(a) > np.mean(b) else model_names[j]
                        print(f"    Significant difference! {better} is better.")
                    else:
                        print("    No significant difference.")

def main():
    # Load data with updated paths
    print("Loading data...")
    queries_json = json.load(open("cranfield/cran_queries.json", 'r'))[:]
    query_ids = [item["query number"] for item in queries_json]
    queries = [item["query"] for item in queries_json]
    
    docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
    doc_ids = [item["id"] for item in docs_json]
    docs = [item["body"] for item in docs_json]
    
    qrels = json.load(open("cranfield/cran_qrels.json", 'r'))[:]
    
    # Initialize LSA engine
    lsa_engine = LSASearchEngine(n_components=300)
    
    # Find optimal number of components
    print("\nFinding optimal number of components...")
    optimal_components = lsa_engine.find_optimal_components(
        docs, doc_ids, queries, query_ids, qrels,
        min_components=100, max_components=1500, step=100
    )
    
    # Build index with optimal number of components
    print(f"\nBuilding LSA index with {optimal_components} components...")
    lsa_engine.n_components = optimal_components
    lsa_engine.svd = TruncatedSVD(n_components=optimal_components, random_state=42)
    lsa_engine.build_index(docs, doc_ids)
    
    # Standard LSA
    print("\nEvaluating standard LSA...")
    doc_IDs_ordered_lsa = lsa_engine.search(queries)
    results_lsa = lsa_engine.evaluate_performance(doc_IDs_ordered_lsa, query_ids, qrels)
    
    # POS-tagged LSA
    print("\nEvaluating POS-tagged LSA...")
    lsa_engine.build_index(docs, doc_ids, use_pos_tagging=True)
    doc_IDs_ordered_pos_lsa = lsa_engine.search(queries, use_pos_tagging=True)
    results_pos_lsa = lsa_engine.evaluate_performance(doc_IDs_ordered_pos_lsa, query_ids, qrels)
    
    # Extract per-query metrics for comparison
    query_metrics_lsa = {'Precision': [], 'Recall': [], 'F-score': [], 'nDCG': []}
    query_metrics_pos_lsa = {'Precision': [], 'Recall': [], 'F-score': [], 'nDCG': []}
    
    eval_metrics = Evaluation()
    k = 10  # Using k=10 for comparison
    
    # Get per-query metrics for standard LSA
    for i, query_id in enumerate(query_ids):
        true_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(query_id)]
        query_docs = doc_IDs_ordered_lsa[i][:k]
        
        query_metrics_lsa['Precision'].append(eval_metrics.queryPrecision(query_docs, query_id, true_docs, k))
        query_metrics_lsa['Recall'].append(eval_metrics.queryRecall(query_docs, query_id, true_docs, k))
        query_metrics_lsa['F-score'].append(eval_metrics.queryFscore(query_docs, query_id, true_docs, k))
        query_metrics_lsa['nDCG'].append(eval_metrics.queryNDCG(query_docs, query_id, true_docs, k))
    
    # Get per-query metrics for POS-tagged LSA
    for i, query_id in enumerate(query_ids):
        true_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(query_id)]
        query_docs = doc_IDs_ordered_pos_lsa[i][:k]
        
        query_metrics_pos_lsa['Precision'].append(eval_metrics.queryPrecision(query_docs, query_id, true_docs, k))
        query_metrics_pos_lsa['Recall'].append(eval_metrics.queryRecall(query_docs, query_id, true_docs, k))
        query_metrics_pos_lsa['F-score'].append(eval_metrics.queryFscore(query_docs, query_id, true_docs, k))
        query_metrics_pos_lsa['nDCG'].append(eval_metrics.queryNDCG(query_docs, query_id, true_docs, k))
    
    # Compare standard LSA vs POS-tagged LSA
    print("\nComparing Standard LSA vs POS-tagged LSA...")
    lsa_engine.compare_models(
        [query_metrics_lsa, query_metrics_pos_lsa],
        ["Standard LSA", "POS-tagged LSA"]
    )
    
    # Save results to a file
    results = {
        'Standard LSA': results_lsa,
        'POS-tagged LSA': results_pos_lsa,
        'Optimal Components': optimal_components
    }
    
    with open('lsa_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAnalysis complete. Results saved to lsa_results.json")

if __name__ == "__main__":
    main()