from util import *

import math




class Evaluation():

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

		precision = -1

		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in true_doc_IDs]
		if k > 0:
			precision = len(relevant_retrieved) / k
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
			true_doc_IDs = [rel['id'] for rel in qrels if rel['query_num'] == query_id]
			p = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(p)
		return sum(precisions) / len(precisions)

	
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

		recall = 0
		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in true_doc_IDs]
		if len(true_doc_IDs) > 0:
			recall = len(relevant_retrieved) / len(true_doc_IDs)
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
			true_doc_IDs = [rel['id'] for rel in qrels if rel['query_num'] == query_id]
			r = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(r)
		return sum(recalls) / len(recalls)


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

		p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if p + r == 0:
			return 0
		return 2 * p * r / (p + r)

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
			true_doc_IDs = [rel['id'] for rel in qrels if rel['query_num'] == query_id]
			f = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(f)
		return sum(fscores) / len(fscores)
	

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

		def dcg(relevance_list):
			return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_list))

		retrieved_k = query_doc_IDs_ordered[:k]
		relevance_scores = [1 if doc_id in true_doc_IDs else 0 for doc_id in retrieved_k]
		idcg_scores = sorted(relevance_scores, reverse=True)

		dcg_val = dcg(relevance_scores)
		idcg_val = dcg(idcg_scores)
		if idcg_val == 0:
			return 0
		return dcg_val / idcg_val


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
			true_doc_IDs = [rel['id'] for rel in qrels if rel['query_num'] == query_id]
			ndcg = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			ndcgs.append(ndcg)
		return sum(ndcgs) / len(ndcgs)


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

		retrieved_k = query_doc_IDs_ordered[:k]
		hits = 0
		sum_precisions = 0
		for i, doc_id in enumerate(retrieved_k):
			if doc_id in true_doc_IDs:
				hits += 1
				sum_precisions += hits / (i + 1)
		if hits == 0:
			return 0
		return sum_precisions / hits


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
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
			true_doc_IDs = [rel['id'] for rel in q_rels if rel['query_num'] == query_id]
			avg_p = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			avg_precisions.append(avg_p)
		return sum(avg_precisions) / len(avg_precisions)

