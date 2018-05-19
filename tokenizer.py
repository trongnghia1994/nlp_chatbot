import nltk


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def tokenize(sequence):
        return nltk.word_tokenize(sequence)
