# Difference Between Stemming and Lemmatization

##Definition

- **Stemming**: A process of reducing words to their base/root form by removing suffixes using simple rules.  
- **Lemmatization**: A process of converting words into their meaningful base form (lemma) using vocabulary and grammar rules.

## Comparison Table

| Feature | Stemming | Lemmatization |
|--------|---------|--------------|
| Method | Rule-based cutting | Dictionary + grammar-based |
| Output | May not be a real word | Always meaningful word |
| Accuracy | Low | High |
| Speed | Fast | Slower |
| Context Awareness | No | Yes (uses POS tagging) |
| Example | running → runn  | running → run  |
| Example | studies → studi  | studies → study  |
| Tools | PorterStemmer | WordNetLemmatizer |
