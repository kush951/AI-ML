import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# 🔹 1. Setup (Download required resources)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 🔹 Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_bio(text):
    """
    Cleans and preprocesses input text using NLP pipeline:
    - Lowercasing
    - Removing punctuation
    - Tokenization
    - Stopword removal
    - Lemmatization
    """

    #Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization (split sentence into words)
    tokens = word_tokenize(text)

    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in tokens if word not in stop_words]

    #Lemmatization (convert words to root form)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]

    #Join words back into sentence
    return " ".join(lemmatized_text)


#Example Input
sample_bio = "I love Hiking in the mountains and Coding late at night!"

# 🔹 Output
print("Original Bio:", sample_bio)
print("Cleaned Bio:", clean_bio(sample_bio))