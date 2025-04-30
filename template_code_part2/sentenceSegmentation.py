from util import *

# Add your import statements here

import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import PunktTokenizer


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		# Split entire text using whitespaces

		split = text.split()

		# Define sentence delimiters

		punctuation_marks = [".", "!", "?"]

		# Pre-defined list of abbreviations (not exhaustive)

		abbreviations = {"mr.", "mrs.", "dr.", "u.s.", "st.", "etc.", "e.g.", "i.e.", "vs.", "prof."}

		# Initialising a sentence as an empty string

		sentence = ""

		for word in split:

			# Adding words to the sentence if it does not end with a delimiter or is an abbreviation

			if word[-1] not in punctuation_marks or (word[-1] in punctuation_marks and word.lower() in abbreviations):
				sentence += word + " "

			# Adding last word to the sentence if it ends with a delimiter, and starting a new sentence

			if word[-1] in punctuation_marks and word.lower() not in abbreviations:
				sentence += word
				segmentedText.append(sentence)
				sentence = ""

		return segmentedText[0]


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		tokenizer = PunktTokenizer()
		segmentedText = tokenizer.tokenize(text.strip())

		return segmentedText[0]
