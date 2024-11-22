class EnsembleModel:
    def __init__(self, vader_model, transformer_model, toxicity_model):
        self.vader_model = vader_model
        self.transformer_model = transformer_model
        self.toxicity_model = toxicity_model

    def predict(self, text):
        """
        Combine predictions from VADER, Transformer, and Toxicity models.
        :param text: Input text.
        :return: Combined sentiment and toxicity scores.
        """
        vader_scores = self.vader_model.predict(text)
        transformer_scores = self.transformer_model.predict(text)
        toxicity_score = self.toxicity_model.predict(text)

        # Weighted average
        combined_score = {
            'negative': (vader_scores['neg'] + transformer_scores['negative']) / 2,
            'neutral': (vader_scores['neu'] + transformer_scores['neutral']) / 2,
            'positive': (vader_scores['pos'] + transformer_scores['positive']) / 2,
            'toxicity': toxicity_score['toxicity']
        }
        return combined_score
