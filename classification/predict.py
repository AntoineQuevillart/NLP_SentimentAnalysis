import joblib
from tools.cleaning import clean_review
from sentence_transformers import SentenceTransformer

# Initialize encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load the model from disk
svm_model = joblib.load(open("C:/model/svm_classifier.joblib", 'rb'))

# Predict function
def predict():
    # Asking the user to type a review
    review = input('Please type a review: ')

    review = clean_review(review)

    embeddings = encoder.encode(review)

    embeddings = embeddings.reshape(1, 384)

    prediction = svm_model.predict(embeddings)

    print(prediction)

if __name__ == '__main__':
    # Running the loop forever
    while True:
        predict()
        break
