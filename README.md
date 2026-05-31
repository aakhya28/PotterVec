# ⚡ PotterVec

PotterVec is an NLP project that trains a custom Word2Vec model from scratch on the Harry Potter book corpus and explores semantic relationships between characters, spells, houses, and magical concepts.

The project demonstrates how word embeddings capture contextual meaning and relationships directly from text without any pre-trained language models.

---

## 🚀 Features

### 🔍 Similarity Search
Find words that are semantically similar to a given word.

Example:

Input:
harry

Output:
ron
hermione
dumbledore

---

### 🧠 Word Analogies

Perform vector arithmetic on words.

Example:

harry - gryffindor + slytherin

Output:
draco

---

### 🎯 Odd-One-Out Detection

Identify the word that does not belong in a semantic group.

Example:

harry ron hermione voldemort

Output:
voldemort

---

### 📊 Embedding Visualization

Visualize learned word embeddings using:

- PCA
- t-SNE

The visualization reveals natural clusters of:

- Hogwarts houses
- Professors
- Students
- Villains
- Magical objects

---

## 🛠️ Tech Stack

- Python
- Gensim
- NLTK
- Scikit-Learn
- Matplotlib
- Streamlit

---

## 📂 Project Structure

PotterVec/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Harry_Potter_all_books_preprocessed.txt

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/aakhya28/PotterVec
cd PotterVec