from util import *

# Add your import statements here

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

import json
from nltk.tokenize.treebank import TreebankWordTokenizer
import statistics

class StopwordRemoval():

	def fromList(self, text):
		"""
		Stop word removal using NLTK stopwords.

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stop_words = set(stopwords.words("english"))

		stopwordRemovedText = []

		filtered_tokens = [token for token in text if token not in stop_words]
		stopwordRemovedText.append(filtered_tokens)

		return stopwordRemovedText[0]


	def fromCorpus(self, text):

		corpus = json.load(open("cranfield\cran_docs.json", 'r'))
		docs = [item["body"] for item in corpus]
		tokenizer = TreebankWordTokenizer()
		tokenizedText = [tokenizer.tokenize(body) for body in docs]
		flat_tokens = []
		for tokenized in tokenizedText:
			flat_tokens += tokenized
		unique_tokens = set(flat_tokens)
		counts = {token: flat_tokens.count(token) for token in unique_tokens}
		mean_freq = statistics.mean(counts.values())
		sd_freq = statistics.stdev(counts.values())
		stop_words = {word for word, freq in counts.items() if freq >= mean_freq + sd_freq}

		stopwordRemovedText = []

		filtered_tokens = [token for token in text if token not in stop_words]
		stopwordRemovedText.append(filtered_tokens)

		return stopwordRemovedText[0]
