import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

import json
from nltk.tokenize.treebank import TreebankWordTokenizer
import statistics

class StopwordRemoval:

    def fromList(self, text):
        """
        Stop word removal using NLTK stopwords.

        Parameters
        ----------
        text : list of list of str
            A list of sentences, where each sentence is a list of word tokens.

        Returns
        -------
        list of list of str
            A list of sentences where stopwords have been removed from each sentence.
        """
        stop_words = set(stopwords.words("english"))
        stopwordRemovedText = []

        # Iterate through each sentence in the text
        for sentence in text:
            # Filter out tokens that are stopwords
            filtered_tokens = [token for token in sentence if token.lower() not in stop_words]
            stopwordRemovedText.append(filtered_tokens)

        return stopwordRemovedText


    def fromCorpus(self, text):
        """
        Stop word removal based on corpus statistics (mean and standard deviation).

        Parameters
        ----------
        text : list of list of str
            A list of sentences, where each sentence is a list of word tokens.

        Returns
        -------
        list of list of str
            A list of sentences where stopwords (determined from corpus frequency) are removed from each sentence.
        """
        # Load corpus and extract tokenized documents
        corpus = json.load(open("cranfield/cran_docs.json", 'r'))
        docs = [item["body"] for item in corpus]
        
        # Tokenize the documents
        tokenizer = TreebankWordTokenizer()
        tokenizedText = [tokenizer.tokenize(body) for body in docs]
        
        # Flatten the list of tokenized text for analysis
        flat_tokens = []
        for tokenized in tokenizedText:
            flat_tokens += tokenized
        
        # Get unique tokens and calculate frequency statistics
        unique_tokens = set(flat_tokens)
        counts = {token: flat_tokens.count(token) for token in unique_tokens}
        mean_freq = statistics.mean(counts.values())
        sd_freq = statistics.stdev(counts.values())
        
        # Determine stopwords based on frequency (tokens with frequency higher than mean + sd)
        stop_words = {word for word, freq in counts.items() if freq >= mean_freq + sd_freq}

        stopwordRemovedText = []

        # Iterate through each sentence in the text
        for sentence in text:
            # Filter out tokens that are in the stop words set
            filtered_tokens = [token for token in sentence if token.lower() not in stop_words]
            stopwordRemovedText.append(filtered_tokens)

        return stopwordRemovedText
