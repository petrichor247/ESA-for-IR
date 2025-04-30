from util import *

# Add your import statements here

from nltk.stem import PorterStemmer



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		stemmer = PorterStemmer()

		reducedText = []

		stemmed_tokens = [stemmer.stem(token) for token in text]
		reducedText.append(stemmed_tokens)

		return reducedText[0]
