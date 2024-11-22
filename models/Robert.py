from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

class TransformerModel:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        """
        Predict sentiment scores using a transformer model.
        :param text: Input text.
        :return: Dictionary of sentiment scores.
        """
        encoded_text = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = softmax(output[0][0].detach().numpy())
        return {'negative': scores[0], 'neutral': scores[1], 'positive': scores[2]}
