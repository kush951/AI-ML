# 🔗 Day 18: Natural Language Processing (NLP)

##  Objective

Transition from structured numerical data to unstructured text data. Learn how to clean, preprocess, vectorize, and analyze human language to extract user interests for intelligent matchmaking.

---

# 🔗 SECTION 1: Turning Words into Numbers

Computers cannot understand text directly. NLP converts text into numerical representations using techniques like:

* Bag of Words (BoW)
* TF-IDF
* Word Embeddings

---

# 🔗 SECTION 2: Text Preprocessing Pipeline

Before analysis, text must be cleaned and standardized.

##  Steps Performed

* Lowercasing
* HTML Tag Removal
* URL Removal
* Punctuation Removal
* Chat Word Treatment
* Spelling Correction
* Stopword Removal
* Emoji Handling
* Tokenization
* Lemmatization

---

##  Example

**Original Text:**

```
"I love Hiking in the mountains and Coding late at night!"
```

**Cleaned Text:**

```
"love hiking mountains coding late night"
```

---

# 🔗 Stemming vs Lemmatization

* **Stemming**

  * Cuts words using rules
  * May produce incorrect words
  * Fast but less accurate

* **Lemmatization**

  * Converts words to meaningful base form
  * Uses dictionary + POS tagging
  * More accurate but slower

---

# 🔗 SECTION 3: Vectorization (TF-IDF)

TF-IDF assigns importance to words based on their frequency and uniqueness.

##  Key Idea

* Common words → low importance
* Rare words → high importance

---

##  Sample Bios

```
1. Expert in Python and Machine Learning for social good.
2. Professional Chef who loves outdoor Hiking and mountains.
3. Machine Learning enthusiast and mountain hiker.
```

---

##  Output

* Vocabulary Size: 19
* Matrix Shape: (3, 19)

---

# 🔗 Cosine Similarity (Matchmaking Engine)

Cosine similarity measures similarity between user profiles.

##  Result

```
[[1.00, 0.0469, 0.2589],
 [0.0469, 1.00, 0.0606],
 [0.2589, 0.0606, 1.00]]
```

---

##  Observations

* Highest similarity: User 0 & User 2
* User 1 has low similarity with others
* Matching based on shared keywords
* System successfully identifies similar interests

---

# 🔗 SECTION 4: Sentiment Analysis

Used TextBlob to analyze user reviews.

##  Metrics

* **Polarity** (-1 to +1)
* **Subjectivity** (0 to 1)

---

## Insight

* Positive reviews → good event quality
* Negative reviews → poor user experience

---

# 🔗 SECTION 5: Word Embeddings (Word2Vec)

Word2Vec converts words into dense vector representations capturing semantic meaning.

---

##  CBOW (Continuous Bag of Words)

* Context → Target word
* Faster training
* Works well on large datasets

---

##  Skip-Gram

* Word → Context
* Slower training
* Better for small datasets and rare words

---

# ◈ Model Outputs & Analysis

## 🔹 Word Vector Example

```
Vector for 'language':
[-0.0047, -0.0043, -0.0048, -0.0098, -0.0007, ...]
```

👉 Each word is represented as a multi-dimensional numerical vector.

---

## 🔗 Similar Words (CBOW)

* "processing" → powerful, fox, lazy, quick, analysis
* "ai" → machine, sentiment, include

👉 Observation:

* Some irrelevant words appear due to small dataset.

---

## 🔗 Similar Words (Skip-Gram)

* Results almost identical to CBOW

 Reason:

* Limited training data
* Model could not learn strong semantic relationships

---

# 🔗 Word Arithmetic (Semantic Relationships)

Word2Vec supports vector-based reasoning:

## Examples

```
'nlp' - 'language' + 'machine' → 'fascinating'
'deep' - 'learning' + 'machine' → 'embeddings'
```

 Insight:

* Demonstrates ability to capture relationships
* Results are approximate due to small dataset

---

# 🔗 CBOW vs Skip-Gram Comparison

| Feature    | CBOW | Skip-Gram |
| ---------- | ---- | --------- |
| Speed      | Fast | Slow      |
| Small Data | Weak | Better    |
| Accuracy   | Good | Better    |

 In this experiment:
Both models produced similar outputs due to limited dataset.

---

# 🔗 Limitations

* Very small dataset
* Weak semantic understanding
* Noisy similarity results

---

# 🔗 AI Pro Tip

* TF-IDF → importance
* Word2Vec → meaning
* Transformers (BERT, GPT) → context understanding

---

# 🔗Conclusion

* NLP enables understanding of user interests
* TF-IDF + Cosine Similarity powers matchmaking
* Sentiment Analysis improves experience
* Word2Vec captures semantic meaning between words

---

# 🔗Final Takeaway

NLP transforms raw text into meaningful insights, enabling intelligent systems like MeetMux to match users based on interests and emotions.

---
