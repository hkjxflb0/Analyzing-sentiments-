import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text):
        """
        Predict sentiment scores using VADER.
        :param text: Input text.
        :return: Dictionary of sentiment scores (neg, neu, pos, compound).
        """
        return self.analyzer.polarity_scores(text)
