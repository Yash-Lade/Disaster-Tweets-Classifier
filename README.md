# Disaster Tweets Classifier 

This repository contains a machine learning project that classifies whether a tweet is about a real disaster or not. It uses natural language processing (NLP) techniques and a trained ML model to make predictions.

## 📌 Project Objective

To build a classifier that can accurately identify disaster-related tweets, helping authorities, responders, and organizations prioritize responses in real-time.

## 🔍 Dataset

- **Source**: [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
- **Fields**:
  - `id`: unique identifier for each tweet
  - `text`: the content of the tweet
  - `target`: 1 if the tweet refers to a real disaster, 0 if not

## 🧪 Features

- Text cleaning and preprocessing (removing URLs, stopwords, etc.)
- Tokenization and vectorization (TF-IDF or CountVectorizer)
- Machine Learning model (e.g., Logistic Regression, Random Forest, etc.)
- Streamlit web interface for real-time tweet classification

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / spaCy (for NLP)
- Streamlit (for frontend app)
- Jupyter Notebook (for model development and testing)

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Yash-Lade/Disaster-Tweets-Classifier.git
cd Disaster-Tweets-Classifier
```

2. (Optional) Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## 🎯 Output

- Enter a tweet in the web interface.
- The model will classify whether it is **Disaster** or **Not Disaster**.

## 📊 Model Performance

The model is evaluated using accuracy, precision, recall, and F1-score.

## 🙌 Contributions

Feel free to fork the repo, make improvements, and create pull requests!

## 📄 License

This project is licensed under the MIT License.

---

**Author:** [Yash Lade](https://github.com/Yash-Lade)
