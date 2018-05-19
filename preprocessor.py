from nltk.stem.lancaster import LancasterStemmer


class Preprocessor:
    _stemmer = LancasterStemmer()

    @staticmethod
    def stem(word):
        return Preprocessor._stemmer.stem(word)
