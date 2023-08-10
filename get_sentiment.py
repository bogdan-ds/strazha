from typing import Tuple
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from preprocess import preprocess_csv


model_id = "rmihaylov/roberta-base-sentiment-bg"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)


def predict_sentiment(text: str) -> Tuple[str, float]:
    # Tokenize the text without truncation
    tokens = tokenizer.encode(text, padding=True,
                              add_special_tokens=False,
                              truncation=False,
                              return_tensors='pt')

    # If the text is shorter than 512 tokens, process as usual
    if tokens.shape[1] <= 512:
        attention_mask = torch.ones(tokens.shape, dtype=torch.long)
        with torch.no_grad():
            outputs = model(tokens, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        prediction = torch.argmax(probs).item()
        probs = torch.max(probs).item()
        return classify_sentiment(prediction), probs

    # For longer texts, use sliding window approach
    else:
        window_size = 510  # 512 minus [CLS] and [SEP] tokens
        chunks = [tokens[0, i:i+window_size]
                  for i in range(0, tokens.shape[1], window_size)]

        chunk_predictions = []
        avg_probs = torch.zeros((1, model.config.num_labels))

        for chunk in chunks:
            # Add [CLS] and [SEP] tokens
            chunk = torch.cat([torch.tensor([101]),
                               chunk, torch.tensor([102])]).unsqueeze(0)
            attention_mask = torch.ones(chunk.size(), dtype=torch.long)
            with torch.no_grad():
                outputs = model(chunk,
                                attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            avg_probs += probs
            prediction = torch.argmax(probs).item()
            chunk_predictions.append(prediction)

        # For the final prediction, take the mode of chunk predictions
        final_prediction = max(chunk_predictions, key=chunk_predictions.count)

        avg_probs /= len(chunks)
        avg_probs = torch.max(avg_probs).item()

        return classify_sentiment(final_prediction), avg_probs


def classify_sentiment(prediction: int) -> str:
    if prediction == 0:
        return 'negative'
    elif prediction == 1:
        return 'positive'


df = preprocess_csv('statements-48ns.csv')

df[['SENTIMENT', 'PROBABILITY']] = df['ТЕКСТ'].apply(
    lambda text: pd.Series(predict_sentiment(text)))


df.to_csv('sentiments_full2.csv', index=False)
