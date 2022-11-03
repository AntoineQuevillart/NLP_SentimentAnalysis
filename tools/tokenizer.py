import numpy as np
import pandas as pd
from cleaning import clean_review

# Reading the dataset
path = 'C:/dataset/video_game.xlsx'
data = pd.read_excel(path)
data = data.dropna()
data = data.head(50000)
data = data.loc[data['reviewText']!=0,:]

# Length of the array
print('There are total {} reviews in the dataset'.format(len(data['reviewText'])))

# Apply the cleaning function
data['reviewText'] = data['reviewText'].apply(clean_review)

# We only need the reviews column
reviews = data['reviewText'].values

# Tokenize words function
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

tokenized_reviews = []
for s in range(len(reviews)):
    tokenized_reviews.append(tokenize_text(reviews[s]))
    if s % 1000 == 0:
        print('--Processing {}th review--'.format(s))

tokenized_reviews = np.array(tokenized_reviews, dtype = object)

# Associate review with note
df = pd.DataFrame(reviews, data["overall"])

# Save tokenized reviews
df.to_pickle('C:/dataset/tokenized_reviews')

