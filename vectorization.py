import pickle
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn import utils
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Vectorization with Doc2Vec - Distrubuted Bag-of-Words & Distributed Memory
reviews = pd.read_pickle('C:/dataset/tokenized_reviews')
reviews = reviews.reset_index()
reviews.columns = ['overall','tokenized_reviews']

# Split train / test (80/20)
train, test = train_test_split(reviews, test_size=0.2, random_state=42)

# Add tag ('overall') to our data
train_tagged = train.apply(
    lambda r:
    TaggedDocument(words=r['tokenized_reviews'], tags=[r.overall]), axis=1)
test_tagged = test.apply(
    lambda r:
    TaggedDocument(words=r['tokenized_reviews'], tags=[r.overall]), axis=1)

# Free up memory
del reviews

# Use the capacity of your processor
cores = multiprocessing.cpu_count()

# Distributed Bag-of-Words method (DBOW)
model_dbow = Doc2Vec(dm=0,
                     vector_size=150,
                     negative=5,
                     hs=0,
                     sample=0,
                     workers=cores)

model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                     total_examples=len(train_tagged.values),
                     epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


# Distributed Memory method (DM)
model_dmm = Doc2Vec(dm=1,
                    dm_mean=1,
                    vector_size=150,
                    window=10,
                    negative=5,
                    min_count=1,
                    workers=cores,
                    alpha=0.065,
                    min_alpha=0.065)

model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                    total_examples=len(train_tagged.values),
                    epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha


# Concatenate DBOW & DM
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
d2v_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])


# Build the vectorization function

def vector_func(model, tagged_docs):
    docs = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0],
                                 model.infer_vector(doc.words)) for doc in docs])
    return targets, regressors

# Apply function
y_train, X_train = vector_func(d2v_model, train_tagged)
y_test, X_test = vector_func(d2v_model, test_tagged)

# Save as a tupple
vectorized_data = (X_train, y_train, X_test, y_test)
pickle.dump(vectorized_data, open("C:/dataset/vectorized_data.pkl", 'wb'))

# Save final model
d2v_model.save('C:\model\d2v.model')
# Load final model
d2v_model = Doc2Vec.load('C:\model\d2v.model')




