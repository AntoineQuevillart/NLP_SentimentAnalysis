import pandas as pd
import pickle
import numpy as np
from tools.cleaning import clean_review
from sentence_transformers import SentenceTransformer

# Reading the dataset
path = 'C:/dataset/video_game.xlsx'
dataset = pd.read_excel(path)
dataset = dataset.dropna()
dataset = dataset.loc[dataset['reviewText']!=0,:]

# Resampling
df1 = dataset.loc[dataset['overall']==5].sample(13661)
df2 = dataset.loc[dataset['overall']==4].sample(13661)
df3 =dataset.loc[dataset['overall']==3].sample(13661)
df4 =dataset.loc[dataset['overall']==1].sample(13661)
df5 = dataset.loc[dataset['overall']==2]
dataset = pd.concat([df1,df2,df4,df5], axis=0) # Remove df3
dataset = dataset.reset_index(drop=True)

# Clean up memory
del df1, df2, df4, df5

# Apply cleaning function over train/test reviews
dataset['reviewText'] = dataset['reviewText'].apply(clean_review)

# Assign reviews with overall > 3 as positive sentiment and overall >= 3 negative sentiment
dataset['sentiment'] = dataset['overall'].apply(lambda rating : +1 if rating >= 3 else 0)
sentiment = dataset["sentiment"]
sentiment = sentiment.reshape(68195,1)
sentiment = np.array(sentiment)

# Define X (reviewText) & y (sentiment)
reviews = dataset[['reviewText','sentiment']]

# Save reviews and sentiment
with open('C:/dataset/target-sent.pkl', "wb") as fOut:
    pickle.dump({'data': reviews}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# Clean up memory
del dataset

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences are encoded by calling model.encode()
embeddings = model.encode(reviews["reviewText"])

# Load sentences & embeddings from disc
with open('C:/dataset/embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']

# Create an array with target (sentiment) and encoded reviews.
encoded_dataset = np.concatenate((sentiment, stored_embeddings), axis=1)

# Save as a pickle file
pickle.dump(encoded_dataset, open("C:/dataset/encoded_dataset.pkl", 'wb'))


