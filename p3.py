import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📧 Spam Email Classifier (SVM + TF-IDF)")
emails = [
    "Congratulations! You've won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discounts on luxury watches",
    "Get rich quick with this secret method",
    "Hello, How are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don't forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday’s discussion",
    "Please confirm your attendance for the workshop",
    "Let’s finalize the budget proposal by Friday"
]
labels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)
model = LinearSVC(C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")
st.subheader("✉️ Enter Email Text")
user_input = st.text_area("Type your email message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT SPAM")
