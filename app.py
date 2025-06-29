import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kfz-Vertrags-Klassifikation", layout="centered")
st.title("📄 Kfz-Vertrags-Klassifikation mit automatischem Feature-Ranking")

@st.cache_data
def load_data():
    df = pd.read_csv("Head_data_kfz_vertrag.csv")
    df = df.dropna(subset=["target"])
    return df

def train_model(df, top_n=10):
    X = df.drop("target", axis=1)
    y = df["target"]

    # Nur numerische und kategoriale Features
    X = X.select_dtypes(include=["object", "int64", "float64"])
    X = X.dropna(axis=1, thresh=0.9 * len(X))  # Drop columns with too many missing values

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), categorical_cols)
    ])

    pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
    pipeline.fit(X, y)

    # Feature Importance holen
    model = pipeline.named_steps["randomforestclassifier"]
    feature_names = pipeline.named_steps["columntransformer"].get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    return pipeline, importance_df

# Daten laden und Modell trainieren
data = load_data()
model, top_features_df = train_model(data, top_n=10)

st.subheader("📊 Wichtigste Merkmale laut Modell")
fig, ax = plt.subplots()
top_features_df.plot.barh(x="feature", y="importance", ax=ax, legend=False)
plt.gca().invert_yaxis()
st.pyplot(fig)

# Eingabemaske
st.subheader("🔍 Vertragsdaten eingeben")

# Spaltennamen extrahieren
selected_features = top_features_df["feature"].tolist()

# Eingabe-Felder aufbauen (vereinfachte Namensrekonstruktion)
input_data = {}
for feat in selected_features:
    base = feat.split("__")[0] if "__" in feat else feat
    col_data = data[base] if base in data.columns else None
    if col_data is not None:
        if pd.api.types.is_numeric_dtype(col_data):
            val = st.number_input(f"{base}", value=float(col_data.median()))
        else:
            val = st.selectbox(f"{base}", sorted(col_data.dropna().unique()))
        input_data[base] = val

# Eingabe vorbereiten als DataFrame
input_df = pd.DataFrame([input_data])

if st.button("🧠 Vorhersage starten"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Der Vertrag wird voraussichtlich angenommen.")
    else:
        st.error("❌ Der Vertrag wird voraussichtlich abgelehnt.")
