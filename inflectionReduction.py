from nltk.stem import PorterStemmer

class InflectionReduction:
    def reduce(self, text):
        """
        Apply stemming using PorterStemmer to reduce inflected words to their base/stem form.

        Parameters
        ----------
        text : list of list of str
            A list of sentences, where each sentence is a list of word tokens.

        Returns
        -------
        list of list of str
            A list of sentences where each token is stemmed.
        """
        stemmer = PorterStemmer()
        reducedText = []

        # Iterate over each sentence in the input text
        for sentence in text:
            # Stem each token in the sentence
            stemmed_sentence = [stemmer.stem(token) for token in sentence]
            reducedText.append(stemmed_sentence)

        return reducedText
