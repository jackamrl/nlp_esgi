from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

# Télécharger les ressources NLTK nécessaires (si pas déjà fait)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialiser le stemmer français
stemmer = SnowballStemmer('french')

# Charger les stop words français
try:
    french_stopwords = set(stopwords.words('french'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    french_stopwords = set(stopwords.words('french'))


def preprocess_text(text):
    """
    Preprocesse un texte avec NLTK :
    - Tokenisation
    - Suppression des stop words
    - Stemming
    
    Args:
        text: Texte à traiter
    
    Returns:
        Texte traité (string)
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Tokeniser (séparer les mots)
    # On utilise une regex simple pour extraire les mots
    tokens = re.findall(r'\b\w+\b', text)
    
    # Supprimer les stop words et faire le stemming
    processed_tokens = []
    for token in tokens:
        if token not in french_stopwords and len(token) > 2:  # Ignorer les mots trop courts
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)
    
    # Rejoindre les tokens en une chaîne
    return ' '.join(processed_tokens)


def make_features(df, vectorizer=None):
    """
    Transforme les titres de vidéos en features encodées avec CountVectorizer.
    
    Args:
        df: DataFrame avec colonnes 'video_name' et 'is_comic'
        vectorizer: CountVectorizer existant (pour predict) ou None (pour train)
    
    Returns:
        X: Features encodées (matrice sparse ou array)
        y: Labels (is_comic)
        vectorizer: Le CountVectorizer utilisé (pour sauvegarder)
    """
    y = df["is_comic"]
    
    # Preprocesser les textes avec NLTK
    processed_texts = df["video_name"].apply(preprocess_text)
    
    # Créer ou utiliser le vectorizer existant
    if vectorizer is None:
        # Mode train : créer un nouveau vectorizer
        vectorizer = CountVectorizer(
            lowercase=False,  # Déjà en minuscules après preprocessing
            token_pattern=r'\b\w+\b',  # Pattern pour les mots
            min_df=2,  # Ignorer les mots qui apparaissent dans moins de 2 documents
            max_df=0.95,  # Ignorer les mots qui apparaissent dans plus de 95% des documents
            ngram_range=(1, 2)  # Utiliser des unigrammes et bigrammes
        )
        X = vectorizer.fit_transform(processed_texts)
    else:
        # Mode predict : utiliser le vectorizer existant
        X = vectorizer.transform(processed_texts)
    

    
    return X, y, vectorizer
