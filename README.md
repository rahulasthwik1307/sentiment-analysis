# Sentiment Analysis Tool for Restaurant Reviews

## Overview

This project is an interactive sentiment analysis application designed to classify restaurant reviews as **positive** or **negative**. Using **machine learning** and **natural language processing (NLP)** techniques, the tool provides actionable insights such as sentiment patterns and review distributions.

---

## Features

- 🚀 **Accurate Sentiment Classification**: Distinguishes between positive and negative reviews.
- 📊 **Visualization**: Displays insights like word clouds, class distributions, and frequent words.
- 🔄 **Interactive UI**: User-friendly Streamlit interface for real-time predictions.
- 📈 **Model Comparison**: Evaluate various models' performance with metrics and visualizations.
- 🌐 **Preprocessing**: Handles text cleaning, tokenization, lemmatization, and spelling corrections.

---

## Dataset

The dataset used for training and testing consists of **1000 restaurant reviews** with the following attributes:

- **Review**: Textual review provided by customers.
- **Liked**: Binary label indicating sentiment (`1` for positive, `0` for negative).

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - NLP: `nltk`, `TfidfVectorizer`
  - Machine Learning: `sklearn`, `joblib`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - Web Interface: `streamlit`
- **Modeling Techniques**:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/sentiment-analysis
   cd sentiment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK resources:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

### Workflow 🔄

1. **Input a Review**: Enter a restaurant review into the provided text box.
2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Remove stopwords, lemmatize words, and correct spelling.
3. **Vectorization**: Transform the preprocessed text using **TF-IDF Vectorizer**.
4. **Model Prediction**:
   - The app uses the trained model to predict sentiment (Positive/Negative).
5. **Insights**:
   - Visualize data trends (word clouds, frequent words, and sentiment distributions).
   - Compare model performance metrics and view ROC curves.

---

## Model Evaluation

- **Best Model**: Logistic Regression
- **Metrics**:
  - Accuracy: `~85%`
  - Cross-Validation Mean: `~84%`

---

## Future Scope

- 🚀 **Advanced NLP Models**: Incorporate transformer-based models like **BERT** for improved accuracy.
- 🌐 **Multilingual Support**: Enable sentiment analysis for reviews in multiple languages.
- 📱 **Integration**: Deploy as a mobile-friendly application or browser extension.
- 📊 **Enhanced Analytics**: Add dashboards for real-time insights and trends.
- 🤝 **User Collaboration**: Allow users to suggest corrections or add feedback to refine the model.

---

## Contribution

Contributions are welcome! Feel free to fork the repository, raise issues, or submit pull requests.

