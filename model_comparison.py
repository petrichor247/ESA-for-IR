# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from informationRetrieval import InformationRetrieval
# from ESA_class import ESAInformationRetrieval
# from LSA import LSASearchEngine
# from evaluation import Evaluation
# from plotting import plot, plot_comp
# from hypothesis_testing import hypothesis_test
# from sentenceSegmentation import SentenceSegmentation
# from tokenization import Tokenization
# from inflectionReduction import InflectionReduction
# from stopwordRemoval import StopwordRemoval

# # Download required NLTK data
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')

# class SearchEngine:
#     def __init__(self):
#         self.tokenizer = Tokenization()
#         self.sentenceSegmenter = SentenceSegmentation()
#         self.inflectionReducer = InflectionReduction()
#         self.stopwordRemover = StopwordRemoval()

#     def preprocess_text(self, text):
#         """Preprocess text using the same pipeline as main.py"""
#         # Segment sentences
#         segmented = self.sentenceSegmenter.punkt(text)
#         # Tokenize
#         tokenized = self.tokenizer.pennTreeBank(segmented)
#         # Reduce inflection
#         reduced = self.inflectionReducer.reduce(tokenized)
#         # Remove stopwords
#         processed = self.stopwordRemover.fromList(reduced)
#         return processed

# def load_data():
#     """Load the Cranfield dataset"""
#     queries_json = json.load(open("cranfield/cran_queries.json", 'r'))
#     docs_json = json.load(open("cranfield/cran_docs.json", 'r'))
#     qrels = json.load(open("cranfield/cran_qrels.json", 'r'))
    
#     query_ids = [item["query number"] for item in queries_json]
#     queries = [item["query"] for item in queries_json]
#     doc_ids = [item["id"] for item in docs_json]
#     docs = [item["body"] for item in docs_json]
    
#     return queries, query_ids, docs, doc_ids, qrels

# def evaluate_model(model, queries, query_ids, docs, doc_ids, qrels, model_name, k=10):
#     """Evaluate a single model and return its metrics"""
#     # Build index and rank documents
#     if isinstance(model, InformationRetrieval):
#         # VSM model - use SearchEngine preprocessing
#         search_engine = SearchEngine()
#         processed_docs = [search_engine.preprocess_text(doc) for doc in docs]
#         processed_queries = [search_engine.preprocess_text(query) for query in queries]
#         model.buildIndex(processed_docs, doc_ids)
#         doc_IDs_ordered = model.rank(processed_queries)
#     elif isinstance(model, ESAInformationRetrieval):
#         # ESA model
#         model.build_vectors()
#         doc_IDs_ordered = []
#         for query in queries:
#             ranked_docs = model.rank(query)
#             doc_IDs_ordered.append(ranked_docs)
#     else:
#         # LSA model
#         model.build_index(docs, doc_ids)
#         doc_IDs_ordered = model.search(queries)
    
#     # Get evaluation metrics
#     evaluator = Evaluation()
#     metrics = {
#         'Precision': [],
#         'Recall': [],
#         'F-score': [],
#         'nDCG': []
#     }
    
#     print(f"\nEvaluating {model_name} model:")
#     print("\nAnalyzing document rankings:")
    
#     # Analyze ranking similarities
#     ranking_similarities = []
#     for i in range(len(doc_IDs_ordered)):
#         for j in range(i+1, len(doc_IDs_ordered)):
#             # Calculate Jaccard similarity between top-k rankings
#             set1 = set(doc_IDs_ordered[i][:k])
#             set2 = set(doc_IDs_ordered[j][:k])
#             similarity = len(set1.intersection(set2)) / len(set1.union(set2))
#             ranking_similarities.append(similarity)
    
#     print(f"Average Jaccard similarity between top-{k} rankings: {np.mean(ranking_similarities):.4f}")
#     print(f"Std dev of Jaccard similarities: {np.std(ranking_similarities):.4f}")
    
#     # Analyze ranking diversity
#     unique_docs_in_top_k = set()
#     for ranking in doc_IDs_ordered:
#         unique_docs_in_top_k.update(ranking[:k])
#     print(f"Number of unique documents in top-{k} across all queries: {len(unique_docs_in_top_k)}")
#     print(f"Total number of documents: {len(doc_ids)}")
    
#     for i, query_id in enumerate(query_ids):
#         true_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(query_id)]
#         query_docs = doc_IDs_ordered[i][:k]
        
#         precision = evaluator.queryPrecision(query_docs, query_id, true_docs, k)
#         recall = evaluator.queryRecall(query_docs, query_id, true_docs, k)
#         fscore = evaluator.queryFscore(query_docs, query_id, true_docs, k)
#         ndcg = evaluator.queryNDCG(query_docs, query_id, true_docs, k)
        
#         metrics['Precision'].append(precision)
#         metrics['Recall'].append(recall)
#         metrics['F-score'].append(fscore)
#         metrics['nDCG'].append(ndcg)
        
#         print(f"\nQuery {query_id}:")
#         print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F-score={fscore:.4f}, nDCG={ndcg:.4f}")
#         print(f"Top {k} documents: {query_docs}")
#         print(f"Relevant documents: {true_docs}")
#         print(f"Relevant docs in top {k}: {[doc for doc in query_docs if doc in true_docs]}")
    
#     print(f"\nMean metrics for {model_name}:")
#     for metric in metrics:
#         mean_val = np.mean(metrics[metric])
#         std_val = np.std(metrics[metric])
#         print(f"{metric}: mean={mean_val:.4f}, std={std_val:.4f}")
    
#     # Plot individual model metrics
#     plot(qrels, doc_IDs_ordered, queries, k, model_name)
    
#     return metrics, doc_IDs_ordered

# def compare_models(model1_metrics, model2_metrics, model1_name, model2_name):
#     """Compare two models using hypothesis testing and plotting"""
#     print(f"\nComparing {model1_name} vs {model2_name}:")
    
#     # Plot comparison
#     plot_comp(
#         model1_metrics['Precision'], model2_metrics['Precision'],
#         model1_metrics['Recall'], model2_metrics['Recall'],
#         model1_metrics['F-score'], model2_metrics['F-score'],
#         model1_metrics['nDCG'], model2_metrics['nDCG'],
#         10, model1_name, model2_name
#     )
    
#     # Hypothesis testing
#     for metric in ['Precision', 'Recall', 'F-score', 'nDCG']:
#         t_stat, p_value = hypothesis_test(model1_metrics[metric], model2_metrics[metric])
#         print(f"\n{metric}:")
#         print(f"t-statistic: {t_stat:.4f}")
#         print(f"p-value: {p_value:.4f}")
#         if p_value < 0.05:
#             better = model1_name if np.mean(model1_metrics[metric]) > np.mean(model2_metrics[metric]) else model2_name
#             print(f"Significant difference! {better} performs better.")
#         else:
#             print("No significant difference.")

# def main():
#     # Load data
#     print("Loading data...")
#     queries, query_ids, docs, doc_ids, qrels = load_data()
    
#     # Initialize models
#     vsm_model = InformationRetrieval()
#     esa_model = ESAInformationRetrieval(corpus_size=10000)  # Using 10000 concepts
#     lsa_model = LSASearchEngine(n_components=300)  # Using 300 components
    
#     # Evaluate each model
#     print("\nEvaluating VSM model...")
#     vsm_metrics, vsm_results = evaluate_model(vsm_model, queries, query_ids, docs, doc_ids, qrels, "VSM")
    
#     print("\nEvaluating ESA model...")
#     esa_metrics, esa_results = evaluate_model(esa_model, queries, query_ids, docs, doc_ids, qrels, "ESA")
    
#     print("\nEvaluating LSA model...")
#     lsa_metrics, lsa_results = evaluate_model(lsa_model, queries, query_ids, docs, doc_ids, qrels, "LSA")
    
#     # Compare models pairwise
#     print("\nPerforming pairwise comparisons...")
#     compare_models(vsm_metrics, esa_metrics, "VSM", "ESA")
#     compare_models(vsm_metrics, lsa_metrics, "VSM", "LSA")
#     compare_models(esa_metrics, lsa_metrics, "ESA", "LSA")
    
#     # Save results
#     results = {
#         'VSM': {metric: np.mean(values) for metric, values in vsm_metrics.items()},
#         'ESA': {metric: np.mean(values) for metric, values in esa_metrics.items()},
#         'LSA': {metric: np.mean(values) for metric, values in lsa_metrics.items()}
#     }
    
#     with open('model_comparison_results.json', 'w') as f:
#         json.dump(results, f, indent=4)
    
#     print("\nAnalysis complete. Results saved to model_comparison_results.json")

# if __name__ == "__main__":
#     main() 

import json
import numpy as np
from informationRetrieval import InformationRetrieval
from ESA_class import ESAInformationRetrieval
from LSA import LSASearchEngine
from evaluation import Evaluation
from plotting import plot, plot_comp
from hypothesis_testing import hypothesis_test
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

class TextPreprocessor:
    def __init__(self):
        self.segmenter = SentenceSegmentation()
        self.tokenizer = Tokenization()
        self.reducer = InflectionReduction()
        self.stopword_remover = StopwordRemoval()

    def preprocess(self, text):
        sentences = self.segmenter.punkt(text)
        tokens = self.tokenizer.pennTreeBank(sentences)
        reduced_tokens = self.reducer.reduce(tokens)
        clean_tokens = self.stopword_remover.fromList(reduced_tokens)
        return clean_tokens

def load_cranfield():
    queries = json.load(open("cranfield/cran_queries.json"))
    docs = json.load(open("cranfield/cran_docs.json"))
    qrels = json.load(open("cranfield/cran_qrels.json"))

    query_texts = [q["query"] for q in queries]
    query_ids = [q["query number"] for q in queries]
    doc_texts = [d["body"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    return query_texts, query_ids, doc_texts, doc_ids, qrels

def evaluate_model(model, query_texts, query_ids, doc_texts, doc_ids, qrels, name, k=10):
    evaluator = Evaluation()
    metrics = {'Precision': [], 'Recall': [], 'F-score': [], 'nDCG': []}

    if isinstance(model, InformationRetrieval):
        processor = TextPreprocessor()
        processed_docs = [processor.preprocess(doc) for doc in doc_texts]
        processed_queries = [processor.preprocess(q) for q in query_texts]
        model.buildIndex(processed_docs, doc_ids)
        ranked_docs = model.rank(processed_queries)

    elif isinstance(model, ESAInformationRetrieval):
        model.build_vectors()
        ranked_docs = [model.rank(query) for query in query_texts]

    else:  # LSA
        model.build_index(doc_texts, doc_ids)
        ranked_docs = model.search(query_texts)

    print(f"\nEvaluating {name} model:")

    for i, qid in enumerate(query_ids):
        relevant_docs = [int(item["id"]) for item in qrels if item["query_num"] == str(qid)]
        top_docs = ranked_docs[i][:k]

        p = evaluator.queryPrecision(top_docs, qid, relevant_docs, k)
        r = evaluator.queryRecall(top_docs, qid, relevant_docs, k)
        f = evaluator.queryFscore(top_docs, qid, relevant_docs, k)
        n = evaluator.queryNDCG(top_docs, qid, relevant_docs, k)

        metrics['Precision'].append(p)
        metrics['Recall'].append(r)
        metrics['F-score'].append(f)
        metrics['nDCG'].append(n)

    for metric, values in metrics.items():
        print(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    plot(qrels, ranked_docs, query_texts, k, name)
    return metrics, ranked_docs

def compare_models(metrics1, metrics2, name1, name2):
    print(f"\nComparing {name1} vs {name2}:")

    plot_comp(metrics1['Precision'], metrics2['Precision'],
              metrics1['Recall'], metrics2['Recall'],
              metrics1['F-score'], metrics2['F-score'],
              metrics1['nDCG'], metrics2['nDCG'],
              10, name1, name2)

    for metric in ['Precision', 'Recall', 'F-score', 'nDCG']:
        t_stat, p_val = hypothesis_test(metrics1[metric], metrics2[metric])
        print(f"\n{metric}: t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            better = name1 if np.mean(metrics1[metric]) > np.mean(metrics2[metric]) else name2
            print(f"Significant difference: {better} is better")
        else:
            print("No significant difference")

def main():
    print("Loading Cranfield dataset...")
    queries, query_ids, docs, doc_ids, qrels = load_cranfield()

    vsm = InformationRetrieval()
    esa = ESAInformationRetrieval(corpus_size=10000)
    lsa = LSASearchEngine(n_components=300)

    vsm_metrics, _ = evaluate_model(vsm, queries, query_ids, docs, doc_ids, qrels, "VSM")
    esa_metrics, _ = evaluate_model(esa, queries, query_ids, docs, doc_ids, qrels, "ESA")
    lsa_metrics, _ = evaluate_model(lsa, queries, query_ids, docs, doc_ids, qrels, "LSA")

    compare_models(vsm_metrics, esa_metrics, "VSM", "ESA")
    compare_models(vsm_metrics, lsa_metrics, "VSM", "LSA")
    compare_models(esa_metrics, lsa_metrics, "ESA", "LSA")

    results = {
        'VSM': {m: np.mean(vals) for m, vals in vsm_metrics.items()},
        'ESA': {m: np.mean(vals) for m, vals in esa_metrics.items()},
        'LSA': {m: np.mean(vals) for m, vals in lsa_metrics.items()}
    }
    json.dump(results, open("model_comparison_results.json", "w"), indent=4)
    print("\nSaved results to model_comparison_results.json")

if __name__ == "__main__":
    main()
