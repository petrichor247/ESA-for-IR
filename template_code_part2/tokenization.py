from util import *

# Add your import statements here

from nltk.tokenize.treebank import TreebankWordTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		if type(text) is list:
			tokenizedText = [subtext.split() for subtext in text]
		else:
			tokenizedText = text.split()

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizer = TreebankWordTokenizer()

		if type(text) is list:
			tokenizedText = [tokenizer.tokenize(subtext) for subtext in text]
		else:
			tokenizedText = tokenizer.tokenize(text)

		return tokenizedText