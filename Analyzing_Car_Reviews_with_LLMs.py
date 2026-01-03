import pandas as pd
import torch

from transformers import logging
logging.set_verbosity(logging.WARNING)

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
import evaluate


# Read data 
df = pd.read_csv("data/car_reviews.csv", delimiter=";")
reviews = df['Review'].tolist()
true_labels = df['Class'].tolist()


# Sentiment analyzing
# Load model
sentiment_analyzer = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Perform inference on car reviews
predicted_labels = sentiment_analyzer(reviews)

for review, prediction, label in zip(reviews, predicted_labels, true_labels):
    print(f"Review: {review}\nPrediction: {prediction}\nLabel: {label}\n")

# Load accuracy and F1 metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

references = [1 if label == "POSITIVE" else 0 for label in true_labels]
predictions = [1 if prediction["label"] == "POSITIVE" else 0 for prediction in predicted_labels]

# Caluate accuracy and F1
accuracy_result = accuracy.compute(references=references, predictions=predictions)["accuracy"]
f1_result = f1.compute(references=references, predictions=predictions)["f1"]
print(f"Accuracy : {accuracy_result}, F1 : {f1_result}")


