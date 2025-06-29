import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

st.title("🌸 Iris-Klassifikation mit KI")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

st.sidebar.header("🔧 Eingabewerte einstellen")
sepal_length = st.sidebar.slider("Sepal Length", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()))
petal_length = st.sidebar.slider("Petal Length", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()))
petal_width = st.sidebar.slider("Petal Width", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

X = df.iloc[:, :-1]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction = model.predict(user_input)
prediction_label = iris.target_names[prediction[0]]

st.subheader("🌟 Vorhersage")
st.write(f"Das Modell sagt voraus: **{prediction_label}**")

if st.checkbox("📊 Klassifikationsbericht anzeigen"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
