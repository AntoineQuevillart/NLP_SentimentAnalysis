# Amazon Reviews Rating Prediction

Override stars by attributing a score to a review using NLP.

## Introduction
Sometimes, a customer's review can be unclear, a comment may not reflect the rating (ex. five-stars method). 
This phenomenon can be explained by various reasons such as the fact that this scoring method depends of the client personnality (is a 3/5 score a good score or not ?). 
To override these rating methods, we work on a way to attribute a score based directly on the analysis of a comment. 
This project is based on sentiment analysis using NLP.

## Dataset
For this project, we use a Amazon Review Dataset (see [jmcauley.ucsd.edu](https://jmcauley.ucsd.edu/data/amazon/)). 
Those reviews are associated to a 5-star rating system.
The original dataset contains 142.8 million reviews, it was divided by categories.
In this project, we are using a small fraction of the original dataset. Specifically, we are using the following file containing 231 780 observations (no duplicates).

* `amazon_reviews_us_Video_Games.tsv`

## Model

### Step 1 : Data cleaning

The collected data may contain some noises which can lead our model to be less efficient. Here we want to erase some useless characters in the reviews.
For example, we don't need URL, numbers or special characters. So the use a function to get rid of that. Further, we don't want contractions (such as "won't" : "will not").
We correct contractions to reinforce our training model.

### Step 2 : Text encoding

Text encoding consists in tokenizing textual inputs to transform text to numerical data.

We use the pretrained encoding model 'all-MiniLM-L6-v2' from [Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). This sentence transformer aims to 
map sentences & paragraphs to a 384 dimensional dense vector space. 

Transformers automate word tokenization. It consists to divide strings into lists of substrings 
(e.g. "The cat is at the window" : ["The", "cat", "is", "at", "the", "window"].

### Step 3 : ML Model

SVM model is used for classification. We implement this model with an accuracy score of ~80%. 
SVM classifier from [scikit learn](https://scikit-learn.org/stable/modules/svm.html) is effective in high dimensional
spaces. We use the five-fold cross-validation argument to provide probabilities on class belonging.


## Structure

Each files on this repository correspond to a step. The folder `tools` contains the following files : 
- `cleaning` - preprocess function to clean data. 
- `bert_encoding` - encoding text data to numeric data with sbert.
- `model` - train a classifier and evaluate it. 
- `predict` - prediction function. 
