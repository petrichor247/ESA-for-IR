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

def compare_models(model1_metrics, model2_metrics, model1_name, model2_name):
    """Compare two models using hypothesis testing and plotting"""
    print(f"\nComparing {model1_name} vs {model2_name}:")
    
    # Plot comparison
    plot_comp(
        model1_metrics['Precision'], model2_metrics['Precision'],
        model1_metrics['Recall'], model2_metrics['Recall'],
        model1_metrics['F-score'], model2_metrics['F-score'],
        model1_metrics['nDCG'], model2_metrics['nDCG'],
        10, model1_name, model2_name
    )
    
    # Hypothesis testing with one-tailed t-test
    for metric in ['Precision', 'Recall', 'F-score', 'nDCG']:
        # Perform one-tailed t-test using hypothesis_test function
        t_stat, p_value = hypothesis_test(model1_metrics[metric], model2_metrics[metric])
        print(f"\n{metric}:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value (one-tailed): {p_value:.4f}")
        
        # For one-tailed test with alternative='greater':
        # H0: model1 <= model2
        # H1: model1 > model2
        if p_value < 0.05:
            # If p-value is significant, we reject H0 and conclude model1 > model2
            print(f"Significant difference! {model1_name} performs better than {model2_name}.")
        else:
            # If p-value is not significant, we fail to reject H0
            # This means we can't conclude model1 is better than model2
            print("No significant difference in performance (cannot conclude model1 is better than model2).")
        
        # Print mean values for context
        mean1 = np.mean(model1_metrics[metric])
        mean2 = np.mean(model2_metrics[metric])
        print(f"Mean {metric}: {model1_name}={mean1:.4f}, {model2_name}={mean2:.4f}")

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
