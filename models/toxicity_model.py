from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

class ToxicityModel:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-offensive"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        """
        Predict toxicity score.
        :param text: Input text.
        :return: Toxicity score.
        """
        encoded_text = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_text)
        print(output)
        score = softmax(output[0][0].detach().numpy())[0]  # Offensive score
        return {'toxicity': score}
