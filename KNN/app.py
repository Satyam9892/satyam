import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title("🌦️ KNN Weather Classification App")
X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
st.sidebar.header("Enter Weather Data")
temp = st.sidebar.slider("Temperature (°C)", 20, 40, 26)
humidity = st.sidebar.slider("Humidity (%)", 50, 100, 78)
new_point = np.array([[temp, humidity]])
prediction = knn.predict(new_point)[0]
st.subheader("Prediction Result:")
if prediction == 0:
    st.success("☀️ Predicted Weather: Sunny")
else:
    st.info("🌧️ Predicted Weather: Rainy")
fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny", s=100)
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy", s=100)
ax.scatter(new_point[0, 0], new_point[0, 1], marker='*', s=300, label="New Data")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(alpha=0.3)

# Show plot in Streamlit
st.pyplot(fig)
