from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InformationRetrieval():

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.docVectors = None  # TF-IDF matrix
        self.docIDs = []        # list of document IDs

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the class variables.

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
        self.docIDs = docIDs
        # Flatten each document into a single string
        flattened_docs = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]
        # Compute TF-IDF matrix
        self.docVectors = self.vectorizer.fit_transform(flattened_docs)

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query.

        Parameters
        ----------
        queries : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query.

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query.
        """
        doc_IDs_ordered = []

        # Flatten each query into a single string
        flattened_queries = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
        # Transform queries using the same TF-IDF vectorizer
        queryVectors = self.vectorizer.transform(flattened_queries)

        # Compute cosine similarities between queries and documents
        sim_matrix = cosine_similarity(queryVectors, self.docVectors)

        for row in sim_matrix:
            # Get sorted indices (highest similarity first)
            ranked_indices = np.argsort(-row)
            ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
            doc_IDs_ordered.append(ranked_docIDs)

        return doc_IDs_ordered
