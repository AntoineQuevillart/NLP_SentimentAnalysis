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

    """
    Recupere le commentaire depuis le fichier excel gestion_de_projet_formulaire.xlsm dans la feuille Input que vous devez créer dans votre machine avec un chemin adapté de type C:/Users/name/Documents/
    Renvoie le score de positivité du commentaire dans un fichier excel output_model.xlsx dans la feuille Output que vous devez créer dans votre machine avec un chemin adapté de type C:/Users/name/Documents/NLP/.

    Note: la fonction utilise la fonction nlp_pipeline, l'encoder de texte, la selection de modèle et le modèle de classifiacation,
    :return:

    Exemple:
        Recuperation : "I love it this game. I only play that since my purchase in 2012, I'm statisfied !"
        Sortie: 0.9
    """

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


