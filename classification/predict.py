import joblib
from tools.cleaning import clean_review
import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load model from disk
svm_model = joblib.load(filename = "C:/dataset/scoring_model.pkl")
selection = joblib.load(filename = 'C:/dataset/select_variable.pkl')

# Predict function
def predict():
    # Asking the user to type a review

    review = pd.read_excel(r"C:/Users/antoi/Documents/python_vba.xlsm", sheet_name="Input")

    review = str(review.dtypes)

    review = clean_review(review)

    embeddings = encoder.encode(review)

    embeddings = embeddings.reshape(1, 384)

    embeddings = pd.DataFrame(embeddings)

    embeddings = selection.transform(embeddings[embeddings.columns[:384]])

    prediction = svm_model.predict_proba(embeddings)

    output = pd.DataFrame(prediction)

    output = round(output, 2)

    output.to_excel(r"C:/Users/antoi/Documents/output_model.xls", sheet_name='Output', index=False)

if __name__ == '__main__':
    while True:
        predict()
        break


