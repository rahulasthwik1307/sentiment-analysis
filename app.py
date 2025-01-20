import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
import shap
from streamlit_lottie import st_lottie

def load_lottie_json(json_file):
    import json
    with open(json_file, 'r') as f:
        return json.load(f)

# Load Lottie animations
loading_animation = load_lottie_json('Animation - 1737345232542.json')

# Load the trained model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [spell.correction(token) for token in tokens if token.isalpha()]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to calculate and display ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for name, data in models.items():
        if hasattr(data['model'], 'predict_proba'):
            y_proba = data['model'].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC: {roc_auc_score(y_test, y_proba):.2f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt)

# Function to plot SHAP values
def plot_shap_values(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:10])
    st.subheader("SHAP Summary Plot")

    # Create a figure and axes explicitly for streamlit
    fig = plt.figure(figsize=(10, 6))  # Adjust the size as needed
    
    # Remove the `ax` argument and directly call `summary_plot()`
    shap.summary_plot(shap_values, X_test[:10], plot_type='bar', show=False)  # No ax argument

    # Use Streamlit's st.pyplot with the figure
    st.pyplot(fig, bbox_inches='tight')

# Streamlit UI
st.title('Sentiment Analysis')
st.write("This app predicts the sentiment of restaurant reviews!")

# Input for user review
user_review = st.text_area("Enter a review")

# Predict Button
if st.button('Predict Sentiment'):
    # Show the loading animation
    st_lottie(loading_animation, height=200, key="loading")

    if user_review:
        # Process the review
        processed_review = preprocess_text(user_review)
        vectorized_review = vectorizer.transform([processed_review]).toarray()
        prediction = model.predict(vectorized_review)
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display the sentiment result
        st.success(f"The sentiment of the review is: **{sentiment}**")

        # Load dataset
        df = pd.read_csv('Restaurant_Reviews 1.tsv', sep='\t')

        # Class Distribution
        st.subheader('Class Distribution')
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Liked', palette='viridis', ax=ax1)
        st.pyplot(fig1)

        # Word Cloud
        st.subheader('Word Cloud of Reviews')
        all_text = ' '.join(df['Review'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # Top 20 Frequent Words
        st.subheader('Top 20 Frequent Words')
        tokens = word_tokenize(all_text.lower())
        tokens = [t for t in tokens if t.isalpha()]
        counter = Counter(tokens)
        top_words = counter.most_common(20)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[w[1] for w in top_words], y=[w[0] for w in top_words], palette='magma', ax=ax2)
        st.pyplot(fig2)

        # Model Evaluation and Comparison
        st.subheader("Model Comparison")
        results = joblib.load("model_results.pkl")  # Load pre-saved model comparison results
        comparison_data = {
            "Model": list(results.keys()),
            "Accuracy": [results[name]['accuracy'] for name in results.keys()],
            "CV Mean": [results[name]['cv_scores'].mean() for name in results.keys()],
            "Training Time (s)": [results[name]['training_time'] for name in results.keys()],
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Plot accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Model", y="Accuracy", data=comparison_df, color='skyblue', ax=ax, label="Accuracy")
        
        # Plot cross-validation mean
        sns.barplot(x="Model", y="CV Mean", data=comparison_df, color='lightgreen', ax=ax, alpha=0.6, label="CV Mean Score")

        # Customize the plot
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Comparison (Accuracy vs Cross-Validation Mean)')
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        # ROC Curve
        st.subheader("ROC Curve for Models")
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')
        plot_roc_curves(results, X_test, y_test)

         # SHAP Values
        X_train = joblib.load('X_train.pkl')
        plot_shap_values(results['Logistic Regression']['model'], X_train, X_test)

    else:
        st.write("Please enter a review to predict sentiment.")
