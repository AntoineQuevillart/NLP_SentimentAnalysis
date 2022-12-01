import re
from nltk.corpus import stopwords

def clean_review(text):
    """
    Renvoie une phrase propre prête à l'encodage.
    C'est-à-dire sans majuscule, sans caractère special, sans chiffre, sans ponctuation mise à part le point et sans les stopwords.

    :param text: string
                Un texte demandé
    :return: string
                Une chaine de caractère sans majuscule, sans caractère special, sans chiffre, sans les stopwords

    Exemple:
        Entrée : clean_review("I love it this game. I only play that since my purchase in 2012, i'm statisfied !")
        Sortie : "love game. play since purchase i'm statisfied "

        Entrée : corpus=["I love it this game.", "I only play that since my purchase in 2012, i'm statisfied "]
                corpus.apply(clean_review)
        Sortie :["love game.", "play since purchase i'm statisfied "]
    """

    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
    text = re.sub(r"(\s\-\s|-$)", "", text)
    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\&", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\£", "", text)
    text = re.sub(r"\%", "", text)
    text = re.sub(r"\:", "", text)
    text = re.sub(r"\@", "", text)
    text = re.sub(r"\-", "", text)

    stopwords_dict = {word: 1 for word in stopwords.words("english")}
    new = ""
    for word in text.split():
        if word not in stopwords_dict:
            new += word
            new += " "
    text = new

    return text