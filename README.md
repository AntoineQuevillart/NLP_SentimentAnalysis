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

### Step 2 : Word Tokenizing

Word tokenization is a small but essential task. It consists to divide strings into lists of substrings 
(e.g. "The cat is at the window" : ["The", "cat", "is", "at", "the", "window"].
Here we are using [nltk Word Tokenizer](https://www.nltk.org/api/nltk.tokenize.html). This step is essential before to apply vectorization.

### Step 3 : Vectorization

In our situtation, we want to transform sentences (reviews) to coordinates. Indeed, algorithms can only deal with numbers (they can't learn directly over words)
The problem is that reviews are constructed with words. So we had to replace those series of words by series of numbers. In other words, we convert sequence into 
vectors of given length (e.g. 150). 

For it, we are using a combination of 2 approach given by [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) models 
(Distributed Bag-of-Words and Distributed Memory).

<p align="center">
<img width="600" height="200" src="https://user-images.githubusercontent.com/114365240/199744377-22931260-b2dc-425b-9e8c-e76b0a3dd9a9.png">
</p>

### Step 4 : ML Model

In this step, we compare several models such as Logistic Regression, KNN or RNN. The best model will be used for prediction over new reviews.

## Structure

Each files on this repository correspond to a step. `cleaning` folder contains the preprocess function to clean data. `tokenizer` file contains the function to split 
sentences into lists of words. `vectorization` file contains the function to convert the train and test data into vectors of length 150. `models` file contains
the differents models tested. 

:soon: A last file will be uploaded to do prediction with a selected model.
