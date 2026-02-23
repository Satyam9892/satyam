import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🎬 Sentiment Analysis (IMDB Reviews)")

# Upload dataset
st.subheader("📂 Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show data
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Convert labels
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['review']
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Prediction
    y_pred = model.predict(X_test_tfidf)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader("📈 Model Accuracy")
    st.write(f"Accuracy: {acc:.2f}")

    # Classification report
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # User input
    st.subheader("✍️ Test Your Own Review")
    user_review = st.text_area("Enter a movie review:")

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review")
        else:
            review_vec = tfidf.transform([user_review])
            prediction = model.predict(review_vec)[0]

            if prediction == 1:
                st.success("😊 Positive Review")
            else:
                st.error("😠 Negative Review")
