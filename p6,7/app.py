import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Title
st.title("🚢 Titanic Survival Prediction")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Titanic Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data preprocessing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Features & target
    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Create 2 columns
    col1, col2 = st.columns(2)

    # LEFT COLUMN → Data + Input
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("🎛 Enter Passenger Details")

        pclass = st.selectbox("Pclass", [1, 2, 3])
        age = st.slider("Age", 0, 80, 25)
        sibsp = st.slider("Siblings/Spouses", 0, 5, 0)
        parch = st.slider("Parents/Children", 0, 5, 0)
        fare = st.slider("Fare", 0, 500, 50)

        sex = st.selectbox("Sex", ["male", "female"])
        embarked = st.selectbox("Embarked", ["Q", "S", "C"])

        predict_btn = st.button("Predict Survival")

    # RIGHT COLUMN → Results
    with col2:
        st.subheader("📈 Model Performance")

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.2f}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # Prediction result
        if predict_btn:
            # Convert input to model format
            sex_male = 1 if sex == "male" else 0
            embarked_q = 1 if embarked == "Q" else 0
            embarked_s = 1 if embarked == "S" else 0

            input_data = [[pclass, age, sibsp, parch, fare,
                           sex_male, embarked_q, embarked_s]]

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.success("🎉 Survived")
            else:
                st.error("💀 Did Not Survive")
