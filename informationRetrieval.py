from util import *
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InformationRetrieval:

    def __init__(self):
        self.index = None
        self.documents = []  # Make sure to store documents in the class

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable.

        Parameters
        ----------
        docs : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document.
        docIDs : list
            A list of integers denoting IDs of the documents.

        Returns
        -------
        None
        """
        inverted_index = {}

        for doc_id, document in zip(docIDs, docs):
            flattened_doc = [token for sentence in document for token in sentence]
            token_counts = Counter(flattened_doc)
            for token, freq in token_counts.items():
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append([doc_id, freq])

        self.index = (inverted_index, len(docs), docIDs)
        self.documents = docs  # Store the documents

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query using TF-IDF and cosine similarity.

        Parameters
        ----------
        queries : list
            A list of queries, where each query is a list of sentences,
            and each sentence is a list of tokens.

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query.
        """
        inverted_index, num_documents, document_IDs = self.index
        vocabulary = list(inverted_index.keys())

        # Prepare the document texts by flattening tokens
        doc_texts = [' '.join(token for sentence in doc for token in sentence) for doc in self.documents]

        # Build a TF-IDF vectorizer with the pre-defined vocabulary from the inverted index
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        tfidf_matrix = vectorizer.fit_transform(doc_texts)  # shape: [num_docs, num_terms]

        ranked_doc_IDs = []

        for query in queries:
            # Flatten the query into a single string of tokens
            query_text = ' '.join(token for sentence in query for token in sentence)

            # Transform query using the same vectorizer (no fitting)
            query_vector = vectorizer.transform([query_text])  # shape: [1, num_terms]

            # Compute cosine similarity between the query and all documents
            similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()

            # Sort documents by similarity score in descending order
            ranked_ids = [doc_id for _, doc_id in sorted(zip(similarities, document_IDs), reverse=True)]

            ranked_doc_IDs.append(ranked_ids)

        return ranked_doc_IDs
