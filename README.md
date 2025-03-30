# Text_Mining

This repository contains Python scripts for sentiment analysis and topic classification using multiple approaches, including Naive Bayes, BERT, and zero-shot classification. It works on various datasets like Dynasent, Amazon Alexa, product reviews, and a custom sentiment-topic dataset.

Features
Sentiment Analysis with Naive Bayes:

Trains a Naive Bayes model using TF-IDF features.

Performs sentiment classification on datasets such as Dynasent, Amazon Alexa, and product reviews.

Sentiment Analysis with BERT:

Uses a pre-trained BERT model to classify sentiment into three categories: positive, neutral, and negative.

Topic Classification with Zero-Shot Classification:

Applies Hugging Face's zero-shot classification pipeline to classify text into predefined topics.

Evaluation:

Model evaluation includes metrics like precision, recall, F1 score, and accuracy.

Confusion matrices are plotted to visualize classification results.

Prerequisites
To run this project, you'll need to install the following Python libraries:

transformers

torch

sklearn

seaborn

matplotlib

pandas

You can install the required libraries by running:

bash
Copy
Edit
pip install transformers torch scikit-learn seaborn matplotlib pandas
Files and Datasets
Sentiment-Topic Test Data: You need to provide a sentiment-topic-test.tsv dataset for testing the models.

Dynasent Dataset: The dataset dynasent-v1.1-round01-yelp-train.jsonl is used for training Naive Bayes on the Dynasent dataset.

Amazon Alexa Data: A dataset of reviews with ratings (amazon_alexa.tsv) is used to classify sentiments based on ratings.

Code Structure
1. Naive Bayes Sentiment Analysis
Dynasent Dataset:

Preprocesses the dataset to remove invalid labels and clean the text.

Applies the Naive Bayes model with TF-IDF vectorization for sentiment classification (positive, negative, neutral).

Amazon Alexa Dataset:

Similar to the Dynasent dataset, but uses ratings to assign sentiments (negative, neutral, positive).

All Product Reviews Dataset:

Applies sentiment classification to product reviews using the same Naive Bayes model.

2. BERT Sentiment Classification
BERT Model:

Uses the pre-trained cardiffnlp/twitter-roberta-base-sentiment-latest model to classify sentiment into three categories: positive, neutral, and negative.

Provides detailed results, including a comparison of predicted versus actual sentiments.

3. Evaluation
Confusion Matrix:

Visualizes classification results for sentiment predictions.

Metrics:

Computes precision, recall, F1 score, and accuracy for all models.

4. Topic Classification (Zero-Shot Classification)
Uses Hugging Face’s zero-shot classification pipeline to classify text into topics.


