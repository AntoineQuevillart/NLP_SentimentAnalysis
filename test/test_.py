import re
import numpy as np
from nltk.corpus import stopwords
from tools.cleaning import clean_review
from classification.predict import predict

texte = "I love it this game. I only play that since my purchase in 2012, I'm statisfied !"


def test_clean_review():
    """
    La fonction teste la fiablité de la fonction clean_review en cas de changement de version!
    """
    assert isinstance(clean_review(texte), str)

    text = clean_review(texte)
    stopwords_dict = {word: 1 for word in stopwords.words("english")}
    caractere = ["$", "*", "-", "?", "!", "/", "|", "#", "&", "£", "%", ":", "@"]
    nombre = np.arange(10)
    if __debug__:
        for word in text.split():
            if word in stopwords_dict:
                raise AssertionError(word)
            if word == word.upper():
                raise AssertionError(word)
            if isinstance(word, int):
                raise AssertionError(word)
            for i in range(len(caractere)):
                if caractere[i] in word:
                    raise AssertionError(word)
            for i in range(len(nombre)):
                if str(nombre[i]) in word:
                    raise AssertionError(word)


def test_predict():
    """
    La fonction teste la fiabilité de la fonction predict en cas de changement de version !
    """
    assert isinstance(predict(), float)
    assert isinstance(predict(), int)