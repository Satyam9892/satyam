import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.title("🩺 Diabetes Prediction (Linear Regression)")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.subheader("📌 Dataset Info")
    st.write(f"Samples: {X.shape[0]}")
    st.write(f"Features: {X.shape[1]}")
with col2:
    st.subheader("📈 Visualization")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(y_test, y_pred, alpha=0.5)
    axs[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "k--",
        lw=2
    )
    axs[0].set_title("True vs Predicted")
    axs[0].set_xlabel("True Values")
    axs[0].set_ylabel("Predicted Values")
    axs[0].grid(True)

    # Plot 2: BMI vs Prediction
    axs[1].scatter(x_test[:, 2], y_pred, alpha=0.7)
    axs[1].set_title("BMI vs Predicted")
    axs[1].set_xlabel("BMI Feature")
    axs[1].set_ylabel("Predicted Progression")
    axs[1].grid(True)

    plt.tight_layout()

    st.pyplot(fig)
