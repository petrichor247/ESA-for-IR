import numpy as np
import math


class Evaluation():

    def get_true_doc_IDs(self, query_id, qrels):
        return [int(item['id']) for item in qrels if int(item['query_num']) == query_id]

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """
        top_k = query_doc_IDs_ordered[:k]
        relevant_retrieved = len([doc for doc in top_k if doc in true_doc_IDs])
        precision = relevant_retrieved / k if k > 0 else 0.0
        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """
        precisions = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = self.get_true_doc_IDs(query_id, qrels)
            precision = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            precisions.append(precision)
        meanPrecision = np.mean(precisions) if precisions else 0.0
        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """
        top_k = query_doc_IDs_ordered[:k]
        relevant_retrieved = len([doc for doc in top_k if doc in true_doc_IDs])
        recall = relevant_retrieved / len(true_doc_IDs) if true_doc_IDs else 0.0
        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """
        recalls = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = self.get_true_doc_IDs(query_id, qrels)
            recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            recalls.append(recall)
        meanRecall = np.mean(recalls) if recalls else 0.0
        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        if precision + recall == 0:
            fscore = 0.0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)
        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """
        fscores = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = self.get_true_doc_IDs(query_id, qrels)
            fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            fscores.append(fscore)
        meanFscore = np.mean(fscores) if fscores else 0.0
        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        def dcg(rel_list):
            return sum([(2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(rel_list)])

        top_k = query_doc_IDs_ordered[:k]
        relevance_scores = [1 if doc in true_doc_IDs else 0 for doc in top_k]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)

        dcg_val = dcg(relevance_scores)
        idcg_val = dcg(ideal_relevance_scores)

        nDCG = dcg_val / idcg_val if idcg_val != 0 else 0.0
        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """
        ndcgs = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = self.get_true_doc_IDs(query_id, qrels)
            ndcg = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            ndcgs.append(ndcg)
        meanNDCG = np.mean(ndcgs) if ndcgs else 0.0
        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """
        top_k = query_doc_IDs_ordered[:k]
        relevant_docs = 0
        precision_sum = 0.0

        for idx, doc in enumerate(top_k):
            if doc in true_doc_IDs:
                relevant_docs += 1
                precision_sum += relevant_docs / (idx + 1)

        avgPrecision = precision_sum / len(true_doc_IDs) if true_doc_IDs else 0.0
        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """
        avg_precisions = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = self.get_true_doc_IDs(query_id, qrels)
            avg_prec = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            avg_precisions.append(avg_prec)
        meanAveragePrecision = np.mean(avg_precisions) if avg_precisions else 0.0
        return meanAveragePrecision
