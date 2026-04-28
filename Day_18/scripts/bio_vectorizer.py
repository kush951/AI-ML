import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Sample Dataset of MeetMux User Bios
bios = [
    "Expert in Python and Machine Learning for social good.",
    "Professional Chef who loves outdoor Hiking and mountains.",
    "Machine Learning enthusiast and mountain hiker."
]

#Initialize TF-IDF Vectorizer
#Converts text → weighted numerical vectors
vectorizer = TfidfVectorizer()

#Fit and Transform
# Learns vocabulary + computes TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(bios)

#View Features (Words)
print("Vocabulary:", vectorizer.get_feature_names_out())

#Shape of matrix (rows = documents, columns = words)
print("Vector Shape:", tfidf_matrix.toarray().shape)

#View actual TF-IDF values
print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())