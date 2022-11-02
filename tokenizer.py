import pickle
import pandas as pd
from cleaning import clean_review
from gensim.models import Doc2Vec

# Reading the dataset
path = 'C:/dataset/video_game.xlsx'
data = pd.read_excel(path)
data = data.head(50000)
data = data.dropna()
data = data.loc[data['reviewText']!=0,:]

# We only need the reviews column
reviews = data['reviewText'].values

# Length of the array
print('There are total {} reviews in the dataset'.format(len(reviews)))

# Apply the cleaning function
data['reviewText'] = data['reviewText'].apply(clean_review)

#10h49

# Saving memory
del reviews
del data

# Saving the cleaned data
with open('C:/dataset/clean_data.xlsx', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import xlrd
reviews.to_excel('C:/dataset/reviews.xlsx', sheet_name='reviews')

from google.colab import files
df.to_csv('output.csv', encoding 'utf-8-sig')
files.download('output.csv')