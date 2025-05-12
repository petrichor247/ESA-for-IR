import pandas as pd
from evaluation import Evaluation
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import os
from datetime import datetime
import sys

# Setup output directory and file
if not os.path.exists('comparison'):
    os.makedirs('comparison')
output_file = os.path.join('comparison', f'comparison_output.txt')

# Redirect stdout to both file and console
class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeOutput(output_file)

def save_plot(plt, filename, model_name=''):
    """Helper function to save plots"""
    if not os.path.exists('comparison/plots'):
        os.makedirs('comparison/plots')
    plt.savefig(os.path.join('comparison/plots', filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot(qrels, ranked_doc_ids, queries, k, model_name=' ', bin_size=20):
    """
    Evaluate a single model and plot distribution of metrics.

    Parameters:
    - qrels: Dictionary of relevant documents per query.
    - ranked_doc_ids: List of lists of ranked documents for each query.
    - queries: List of query strings.
    - k: Top-k results to consider.
    - model_name: Name of the model (used in plot titles).
    - bin_size: Bin count for histograms (optional).
    """
    qrels_df = pd.DataFrame(qrels)
    evaluator = Evaluation()

    perfect_precision, zero_precision = [], []
    perfect_recall, zero_recall = [], []

    precision_scores, recall_scores, fscore_scores, ndcg_scores = [], [], [], []

    print(f"\nEvaluating {model_name} model:")
    print("\nAnalyzing document rankings:")

    for idx, retrieved_docs in enumerate(ranked_doc_ids):
        query_id = idx + 1
        relevant_docs = list(map(int, qrels_df[qrels_df['query_num'] == str(query_id)]['id'].tolist()))

        precision = evaluator.queryPrecision(retrieved_docs, query_id, relevant_docs, k)
        recall = evaluator.queryRecall(retrieved_docs, query_id, relevant_docs, k)
        fscore = evaluator.queryFscore(retrieved_docs, query_id, relevant_docs, k)
        ndcg = evaluator.queryNDCG(retrieved_docs, query_id, relevant_docs, k)

        precision_scores.append(precision)
        recall_scores.append(recall)
        fscore_scores.append(fscore)
        ndcg_scores.append(ndcg)

        print(f"\nQuery {query_id}:")
        print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F-score={fscore:.4f}, nDCG={ndcg:.4f}")
        print(f"Top {k} documents: {retrieved_docs[:k]}")
        print(f"Relevant documents: {relevant_docs}")
        print(f"Relevant docs in top {k}: {[doc for doc in retrieved_docs[:k] if doc in relevant_docs]}")

        if precision == 1:
            perfect_precision.append({'q_id': query_id, 'query': queries[idx],
                                      'rel_docs': relevant_docs, 'ret_docs': retrieved_docs[:k]})
        if precision == 0:
            zero_precision.append({'q_id': query_id, 'query': queries[idx],
                                   'rel_docs': relevant_docs, 'ret_docs': retrieved_docs[:k]})
        if recall == 1:
            perfect_recall.append({'q_id': query_id, 'query': queries[idx],
                                   'rel_docs': relevant_docs, 'ret_docs': retrieved_docs[:k]})
        if recall == 0:
            zero_recall.append({'q_id': query_id, 'query': queries[idx],
                                'rel_docs': relevant_docs, 'ret_docs': retrieved_docs[:k]})

    # Print mean metrics
    print(f"\nMean metrics for {model_name}:")
    for metric_name, scores in [('Precision', precision_scores), ('Recall', recall_scores),
                              ('F-score', fscore_scores), ('nDCG', ndcg_scores)]:
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        print(f"{metric_name}: mean={mean_val:.4f}, std={std_val:.4f}")

    # Plot and save individual metrics
    plot_distribution(precision_scores, 'Precision', model_name, f'{model_name.lower()}_precision.png')
    plot_distribution(recall_scores, 'Recall', model_name, f'{model_name.lower()}_recall.png')
    plot_distribution(fscore_scores, 'F-score', model_name, f'{model_name.lower()}_fscore.png')
    plot_distribution(ndcg_scores, 'nDCG', model_name, f'{model_name.lower()}_ndcg.png')

    return precision_scores, recall_scores, fscore_scores, ndcg_scores, perfect_precision, zero_precision, perfect_recall, zero_recall

def plot_comp(prec1, prec2, rec1, rec2, f1, f2, ndcg1, ndcg2, k, model1=' ', model2=' '):
    """
    Compare metric distributions between two models.

    Parameters:
    - prec1, prec2: Precision lists for Model 1 and Model 2.
    - rec1, rec2: Recall lists for Model 1 and Model 2.
    - f1, f2: F-score lists for Model 1 and Model 2.
    - ndcg1, ndcg2: nDCG lists for Model 1 and Model 2.
    - model1, model2: Names of the models.
    """
    print(f"\nComparing {model1} vs {model2}:")
    
    # Plot and save comparisons
    plot_comparison(prec1, prec2, 'Precision', model1, model2, f'{model1.lower()}_vs_{model2.lower()}_precision.png')
    plot_comparison(rec1, rec2, 'Recall', model1, model2, f'{model1.lower()}_vs_{model2.lower()}_recall.png')
    plot_comparison(f1, f2, 'F-score', model1, model2, f'{model1.lower()}_vs_{model2.lower()}_fscore.png')
    plot_comparison(ndcg1, ndcg2, 'nDCG', model1, model2, f'{model1.lower()}_vs_{model2.lower()}_ndcg.png')

def plot_distribution(metric_values, metric_name, model_name, filename):
    """
    Plot distribution of a single metric using histogram + KDE.

    Parameters:
    - metric_values: List of scores.
    - metric_name: Name of the metric.
    - model_name: Name of the model.
    - filename: Name of the file to save the plot.
    """
    values = np.array(metric_values)
    plt.figure(figsize=(10, 5))
    plt.xlabel(f'{metric_name} @ k')
    plt.ylabel('Query Frequency')
    plt.title(f'{metric_name} Distribution with KDE for {model_name} on Cranfield Dataset')

    if len(np.unique(values)) == 1:
        plt.axvline(x=values[0], color='darkgreen', linestyle='--', label=f'All values = {values[0]:.4f}')
        plt.legend()
        save_plot(plt, filename, model_name)
        return

    try:
        kde = gaussian_kde(values)
        x_vals = np.linspace(min(values), max(values), 1000)
        plt.plot(x_vals, kde(x_vals), color='darkgreen', linewidth=3, label='KDE Curve')
    except np.linalg.LinAlgError:
        print(f"Warning: Could not compute KDE for {metric_name} - values may be too similar")
        plt.hist(values, bins=20, density=True, alpha=0.4, color='orange', edgecolor='black', linewidth=1.2)
        plt.legend(['Histogram only (KDE failed)'])
        save_plot(plt, filename, model_name)
        return

    plt.hist(values, bins=20, density=True, alpha=0.4, color='orange', edgecolor='black', linewidth=1.2)
    plt.legend()
    save_plot(plt, filename, model_name)

def plot_comparison(metric1, metric2, metric_name, model1_name, model2_name, filename):
    """
    Compare two models' metric distributions using histogram + KDE.

    Parameters:
    - metric1, metric2: Lists of metric values from Model 1 and Model 2.
    - metric_name: Metric being compared.
    - model1_name, model2_name: Names of the models.
    - filename: Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.xlabel(f'{metric_name} @ k')
    plt.ylabel('Query Frequency')
    plt.title(f'{metric_name} Comparison: {model1_name} vs {model2_name} (Cranfield Dataset)')

    all_values = np.concatenate([metric1, metric2])
    bins = np.linspace(min(all_values), max(all_values), 20)

    try:
        kde1 = gaussian_kde(metric1)
        kde2 = gaussian_kde(metric2)
        x_vals = np.linspace(min(all_values), max(all_values), 1000)
        plt.plot(x_vals, kde1(x_vals), color='darkred', linewidth=2.5, label=f'{model1_name} KDE')
        plt.plot(x_vals, kde2(x_vals), color='navy', linewidth=2.5, label=f'{model2_name} KDE')
    except np.linalg.LinAlgError:
        print(f"Warning: Could not compute KDE for {metric_name} comparison - values may be too similar")

    plt.hist(metric1, bins=bins, density=True, alpha=0.5, color='lightcoral',
             edgecolor='black', linewidth=1.2, label=model1_name)
    plt.hist(metric2, bins=bins, density=True, alpha=0.5, color='teal',
             edgecolor='black', linewidth=1.2, label=model2_name)

    plt.legend()
    save_plot(plt, filename)
