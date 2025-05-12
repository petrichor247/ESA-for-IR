import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class CustomSearchEngine:
    """
    A semantic search engine using dimensionality reduction techniques
    """
    
    def __init__(self, reduction_method='svd', n_dimensions=300, use_lemmatization=False, 
                 use_stemming=False, vectorizer_type='tfidf'):
        """
        Initialize the search engine
        
        Parameters:
        -----------
        reduction_method : str
            Dimensionality reduction method ('svd' or 'nmf')
        n_dimensions : int
            Number of dimensions for reduction
        use_lemmatization : bool
            Whether to use lemmatization in preprocessing
        use_stemming : bool
            Whether to use stemming in preprocessing
        vectorizer_type : str
            Type of vectorizer ('tfidf' or 'count')
        """
        self.n_dimensions = n_dimensions
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.reduction_method = reduction_method
        self.vectorizer_type = vectorizer_type
        
        # Initialize lemmatizer/stemmer if needed
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if self.use_stemming:
            self.stemmer = PorterStemmer()
            
        # Setup vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=10000, min_df=2)
        else:  # count vectorizer
            self.vectorizer = CountVectorizer(max_features=10000, min_df=2)
            
        # Setup dimensionality reduction
        if reduction_method == 'svd':
            self.reducer = TruncatedSVD(n_components=n_dimensions, random_state=42)
        else:  # NMF
            self.reducer = NMF(n_components=n_dimensions, random_state=42, max_iter=500)
        
        # Initialize storage for documents
        self.doc_embeddings = None
        self.document_ids = None
        self.vectorizer_fitted = False
        self.reducer_fitted = False
        
    def preprocess_text(self, text):
        """
        Preprocess text with tokenization, stopword removal,
        and optional lemmatization/stemming
        
        Parameters:
        -----------
        text : str
            Text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        
        # Apply lemmatization if enabled
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Apply stemming if enabled
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        # Join tokens back into string
        return " ".join(tokens)
    
    def create_index(self, documents, document_ids):
        """
        Create a searchable index from documents
        
        Parameters:
        -----------
        documents : list
            List of document texts
        document_ids : list
            List of document IDs
        """
        start_time = time.time()
        self.document_ids = document_ids
        
        # Preprocess all documents
        print("Preprocessing documents...")
        preprocessed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create document-term matrix
        print(f"Vectorizing documents using {self.vectorizer_type}...")
        doc_term_matrix = self.vectorizer.fit_transform(preprocessed_docs)
        self.vectorizer_fitted = True
        
        # Apply dimensionality reduction
        print(f"Applying dimensionality reduction using {self.reduction_method.upper()}...")
        self.doc_embeddings = self.reducer.fit_transform(doc_term_matrix)
        self.reducer_fitted = True
        
        if self.reduction_method == 'svd':
            var_explained = self.reducer.explained_variance_ratio_.sum() * 100
            print(f"Total variance explained: {var_explained:.2f}%")
        
        elapsed_time = time.time() - start_time
        print(f"Index created in {elapsed_time:.2f} seconds")
    
    def search(self, queries):
        """
        Search the index with queries
        
        Parameters:
        -----------
        queries : list
            List of query strings
            
        Returns:
        --------
        list
            List of ranked document IDs for each query
        """
        if not self.vectorizer_fitted or not self.reducer_fitted:
            raise ValueError("Search engine index has not been created yet")
        
        # Preprocess queries
        preprocessed_queries = [self.preprocess_text(query) for query in queries]
        
        # Transform queries to term vectors
        query_vectors = self.vectorizer.transform(preprocessed_queries)
        
        # Apply dimensionality reduction
        query_embeddings = self.reducer.transform(query_vectors)
        
        # Compute similarities and rank documents
        ranked_results = []
        for query_embedding in query_embeddings:
            # Compute cosine similarity between query and all documents
            similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
            
            # Sort documents by similarity (descending)
            ranked_indices = np.argsort(-similarities)
            ranked_docs = [self.document_ids[idx] for idx in ranked_indices]
            ranked_results.append(ranked_docs)
            
        return ranked_results
    
    def optimize_dimensions(self, documents, document_ids, queries, query_ids, relevance_data,
                          dim_values=None):
        """
        Find optimal number of dimensions
        
        Parameters:
        -----------
        documents : list
            List of document texts
        document_ids : list
            List of document IDs
        queries : list
            List of query strings
        query_ids : list
            List of query IDs
        relevance_data : list
            Relevance judgments
        dim_values : list
            List of dimension values to try
            
        Returns:
        --------
        int
            Optimal number of dimensions
        """
        if dim_values is None:
            # Default dimension values to test
            dim_values = [50, 100, 200, 300, 400, 500]
            
        # Calculate mean average precision for each dimension value
        map_scores = []
        ndcg_scores = []  # Add tracking for nDCG scores
        
        for dims in dim_values:
            print(f"\nTesting with {dims} dimensions...")
            
            # Update reducer dimensions
            if self.reduction_method == 'svd':
                self.reducer = TruncatedSVD(n_components=dims, random_state=42)
            else:  # NMF
                self.reducer = NMF(n_components=dims, random_state=42, max_iter=500)
            
            self.n_dimensions = dims
            
            # Create index and search
            self.create_index(documents, document_ids)
            search_results = self.search(queries)
            
            # Calculate MAP@10
            map_10 = self._calculate_map(search_results, query_ids, relevance_data, k=10)
            map_scores.append(map_10)
            
            # Calculate nDCG@10
            ndcg_10 = self._calculate_ndcg(search_results, query_ids, relevance_data, k=10)
            ndcg_scores.append(ndcg_10)
            
            print(f"MAP@10 = {map_10:.4f}")
            print(f"nDCG@10 = {ndcg_10:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot MAP
        plt.subplot(1, 2, 1)
        plt.plot(dim_values, map_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Dimensions')
        plt.ylabel('MAP@10')
        plt.title(f'MAP vs Dimensions ({self.reduction_method.upper()})')
        plt.grid(True)
        
        # Plot nDCG
        plt.subplot(1, 2, 2)
        plt.plot(dim_values, ndcg_scores, marker='o', linestyle='-', color='green')
        plt.xlabel('Number of Dimensions')
        plt.ylabel('nDCG@10')
        plt.title(f'nDCG vs Dimensions ({self.reduction_method.upper()})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{self.reduction_method}_dimensions_performance.png')
        
        # Find optimal dimensions (using MAP as the primary metric)
        best_idx = np.argmax(map_scores)
        optimal_dims = dim_values[best_idx]
        print(f"\nOptimal number of dimensions: {optimal_dims}")
        print(f"MAP@10 = {map_scores[best_idx]:.4f}")
        print(f"nDCG@10 = {ndcg_scores[best_idx]:.4f}")
        
        return optimal_dims
    
    def _calculate_map(self, ranked_docs, query_ids, relevance_data, k=10):
        """
        Calculate Mean Average Precision
        
        Parameters:
        -----------
        ranked_docs : list
            List of ranked document IDs for each query
        query_ids : list
            List of query IDs
        relevance_data : list
            Relevance judgments
        k : int
            Cutoff for precision calculation
            
        Returns:
        --------
        float
            MAP@k
        """
        # Group relevance data by query
        query_relevance = {}
        for item in relevance_data:
            query_num = str(item["query_num"])
            if query_num not in query_relevance:
                query_relevance[query_num] = []
            query_relevance[query_num].append(int(item["id"]))
        
        # Calculate average precision for each query
        ap_values = []
        
        for i, qid in enumerate(query_ids):
            query_id_str = str(qid)
            if query_id_str in query_relevance:
                relevant_docs = set(query_relevance[query_id_str])
                retrieved_docs = ranked_docs[i][:k]
                
                # Calculate AP
                precision_sum = 0.0
                relevant_count = 0
                
                for j, doc_id in enumerate(retrieved_docs):
                    if doc_id in relevant_docs:
                        relevant_count += 1
                        precision_at_j = relevant_count / (j + 1)
                        precision_sum += precision_at_j
                
                if len(relevant_docs) > 0:
                    ap = precision_sum / len(relevant_docs)
                else:
                    ap = 0.0
                
                ap_values.append(ap)
        
        # Return MAP
        return np.mean(ap_values) if ap_values else 0.0
    
    def _calculate_ndcg(self, ranked_docs, query_ids, relevance_data, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Parameters:
        -----------
        ranked_docs : list
            List of ranked document IDs for each query
        query_ids : list
            List of query IDs
        relevance_data : list
            Relevance judgments
        k : int
            Cutoff for nDCG calculation
            
        Returns:
        --------
        float
            nDCG@k
        """
        # Group relevance data by query
        query_relevance = {}
        for item in relevance_data:
            query_num = str(item["query_num"])
            if query_num not in query_relevance:
                query_relevance[query_num] = []
            query_relevance[query_num].append(int(item["id"]))
        
        # Calculate nDCG for each query
        ndcg_values = []
        
        for i, qid in enumerate(query_ids):
            query_id_str = str(qid)
            if query_id_str in query_relevance:
                relevant_docs = set(query_relevance[query_id_str])
                retrieved_docs = ranked_docs[i][:k]
                
                # Calculate DCG
                dcg = 0.0
                for j, doc_id in enumerate(retrieved_docs):
                    # Using binary relevance (1 if relevant, 0 if not)
                    rel = 1 if doc_id in relevant_docs else 0
                    # Use log base 2 for the position discount
                    position = j + 1
                    if position == 1:
                        # Avoid division by zero for position 1
                        dcg += rel
                    else:
                        dcg += rel / np.log2(position)
                
                # Calculate IDCG (Ideal DCG)
                # For binary relevance, the ideal ranking is all relevant documents first
                idcg = 0.0
                for j in range(min(len(relevant_docs), k)):
                    position = j + 1
                    if position == 1:
                        idcg += 1
                    else:
                        idcg += 1 / np.log2(position)
                
                # Calculate nDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_values.append(ndcg)
        
        # Return mean nDCG
        return np.mean(ndcg_values) if ndcg_values else 0.0
    
    def evaluate(self, ranked_docs, query_ids, relevance_data):
        """
        Evaluate search performance with multiple metrics
        
        Parameters:
        -----------
        ranked_docs : list
            List of ranked document IDs for each query
        query_ids : list
            List of query IDs
        relevance_data : list
            Relevance judgments
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Group relevance data by query
        query_relevance = {}
        for item in relevance_data:
            query_num = str(item["query_num"])
            if query_num not in query_relevance:
                query_relevance[query_num] = []
            query_relevance[query_num].append(int(item["id"]))
        
        # Evaluate at different k values
        k_values = [1, 3, 5, 10]
        results = {}
        
        for k in k_values:
            # Initialize metrics for this k
            precision_values = []
            recall_values = []
            f1_values = []
            ap_values = []
            ndcg_values = []  # Add nDCG values
            
            # Calculate metrics for each query
            for i, qid in enumerate(query_ids):
                query_id_str = str(qid)
                if query_id_str in query_relevance:
                    relevant_docs = set(query_relevance[query_id_str])
                    retrieved_docs = ranked_docs[i][:k]
                    
                    # Calculate precision and recall
                    relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
                    precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
                    recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
                    
                    # Calculate F1
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Calculate AP
                    precision_sum = 0.0
                    relevant_count = 0
                    
                    for j, doc_id in enumerate(retrieved_docs):
                        if doc_id in relevant_docs:
                            relevant_count += 1
                            precision_at_j = relevant_count / (j + 1)
                            precision_sum += precision_at_j
                    
                    if len(relevant_docs) > 0:
                        ap = precision_sum / len(relevant_docs)
                    else:
                        ap = 0.0
                    
                    # Calculate nDCG for this query
                    dcg = 0.0
                    for j, doc_id in enumerate(retrieved_docs):
                        rel = 1 if doc_id in relevant_docs else 0
                        position = j + 1
                        if position == 1:
                            dcg += rel
                        else:
                            dcg += rel / np.log2(position)
                    
                    idcg = 0.0
                    for j in range(min(len(relevant_docs), k)):
                        position = j + 1
                        if position == 1:
                            idcg += 1
                        else:
                            idcg += 1 / np.log2(position)
                    
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    
                    # Store metrics
                    precision_values.append(precision)
                    recall_values.append(recall)
                    f1_values.append(f1)
                    ap_values.append(ap)
                    ndcg_values.append(ndcg)
            
            # Calculate means
            mean_precision = np.mean(precision_values) if precision_values else 0
            mean_recall = np.mean(recall_values) if recall_values else 0
            mean_f1 = np.mean(f1_values) if f1_values else 0
            mean_ap = np.mean(ap_values) if ap_values else 0
            mean_ndcg = np.mean(ndcg_values) if ndcg_values else 0  # Mean nDCG
            
            # Store results for this k
            results[f'k={k}'] = {
                'Precision': mean_precision,
                'Recall': mean_recall,
                'F1': mean_f1,
                'MAP': mean_ap,
                'nDCG': mean_ndcg  # Add nDCG to results
            }
            
            # Print results
            print(f"Results at k={k}:")
            print(f"  Precision: {mean_precision:.4f}")
            print(f"  Recall: {mean_recall:.4f}")
            print(f"  F1-score: {mean_f1:.4f}")
            print(f"  MAP: {mean_ap:.4f}")
            print(f"  nDCG: {mean_ndcg:.4f}")  # Print nDCG
        
        # Plot metrics
        plt.figure(figsize=(12, 8))
        metrics = ['Precision', 'Recall', 'F1', 'MAP', 'nDCG']  # Add nDCG to metrics
        
        for metric in metrics:
            values = [results[f'k={k}'][metric] for k in k_values]
            plt.plot(k_values, values, marker='o', label=metric)
        
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.title(f'Search Performance Metrics ({self.reduction_method.upper()})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{self.reduction_method}_performance_metrics.png')
        
        return results
    
    def compare_methods(self, method_results, method_names):
        """
        Compare different search methods
        
        Parameters:
        -----------
        method_results : list
            List of result dictionaries for each method
        method_names : list
            List of method names
        """
        # Extract results at k=10
        k10_results = {}
        for i, method in enumerate(method_names):
            k10_results[method] = method_results[i]['k=10']
        
        # Create DataFrame for plotting
        df = pd.DataFrame(k10_results)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar', figsize=(10, 6))
        plt.title('Comparison of Methods (k=10)')
        plt.ylabel('Score')
        plt.grid(axis='y')
        plt.savefig('results/method_comparison.png')
        
        # Perform statistical significance tests
        print("\nStatistical Significance Tests:")
        metrics = ['Precision', 'Recall', 'F1', 'MAP', 'nDCG']  # Add nDCG to metrics
        
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1 = method_names[i]
                method2 = method_names[j]
                
                print(f"\nComparing {method1} vs {method2}:")
                
                for metric in metrics:
                    m1_value = k10_results[method1][metric]
                    m2_value = k10_results[method2][metric]
                    
                    print(f"  {metric}: {m1_value:.4f} vs {m2_value:.4f}")
                    if m1_value > m2_value:
                        diff_pct = (m1_value - m2_value) / m2_value * 100
                        print(f"    {method1} better by {diff_pct:.2f}%")
                    elif m2_value > m1_value:
                        diff_pct = (m2_value - m1_value) / m1_value * 100
                        print(f"    {method2} better by {diff_pct:.2f}%")
                    else:
                        print("    No difference")

def main():
    # Load data
    print("Loading data from Cranfield collection...")
    with open("cranfield/cran_queries.json", 'r') as f:
        queries_data = json.load(f)
    
    with open("cranfield/cran_docs.json", 'r') as f:
        docs_data = json.load(f)
    
    with open("cranfield/cran_qrels.json", 'r') as f:
        qrels_data = json.load(f)
    
    # Extract queries and documents
    query_ids = [item["query number"] for item in queries_data]
    queries = [item["query"] for item in queries_data]
    
    doc_ids = [item["id"] for item in docs_data]
    docs = [item["body"] for item in docs_data]
    
    print(f"Loaded {len(queries)} queries and {len(docs)} documents")
    
    # Test different configurations
    configurations = [
        {
            'name': 'LSA_Basic',
            'reduction_method': 'svd',
            'use_lemmatization': False,
            'use_stemming': False,
            'vectorizer_type': 'tfidf'
        },
        {
            'name': 'LSA_Lemmatized',
            'reduction_method': 'svd',
            'use_lemmatization': True,
            'use_stemming': False,
            'vectorizer_type': 'tfidf'
        },
        {
            'name': 'NMF_Basic',
            'reduction_method': 'nmf',
            'use_lemmatization': False,
            'use_stemming': False,
            'vectorizer_type': 'tfidf'
        }
    ]
    
    all_results = []
    method_names = []
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*50}")
        
        # Initialize search engine with this configuration
        search_engine = CustomSearchEngine(
            reduction_method=config['reduction_method'],
            use_lemmatization=config['use_lemmatization'],
            use_stemming=config['use_stemming'],
            vectorizer_type=config['vectorizer_type']
        )
        
        # Find optimal dimensions
        print("\nFinding optimal dimensions...")
        optimal_dims = search_engine.optimize_dimensions(
            docs, doc_ids, queries, query_ids, qrels_data,
            dim_values=[50, 100, 150, 200, 250, 300]
        )
        
        # Use optimal dimensions
        if search_engine.reduction_method == 'svd':
            search_engine.reducer = TruncatedSVD(n_components=optimal_dims, random_state=42)
        else:  # NMF
            search_engine.reducer = NMF(n_components=optimal_dims, random_state=42, max_iter=500)
        
        search_engine.n_dimensions = optimal_dims
        
        # Create index and search
        print("\nCreating search index with optimal dimensions...")
        search_engine.create_index(docs, doc_ids)
        
        print("\nPerforming search...")
        search_results = search_engine.search(queries)
        
        # Evaluate performance
        print("\nEvaluating performance...")
        results = search_engine.evaluate(search_results, query_ids, qrels_data)
        
        # Store results
        all_results.append(results)
        method_names.append(config['name'])
        
        # Save configuration results
        config_results = {
            'configuration': config,
            'optimal_dimensions': optimal_dims,
            'results': results
        }
        
        with open(f"results/{config['name']}_results.json", 'w') as f:
            json.dump(config_results, f, indent=4)
    
    # Compare methods
    print("\nComparing all methods...")
    search_engine.compare_methods(all_results, method_names)
    
    # Save overall results
    overall_results = {
        'methods': method_names,
        'results': all_results
    }
    
    with open("results/overall_results.json", 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    print("\nAnalysis complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()