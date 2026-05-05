import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Prediccion de Riesgo de ACV",
    page_icon="🧠",
    layout="wide"
)

st.title("Sistema de Prediccion de Riesgo de ACV")
st.markdown(
    "Dashboard ejecutivo para apoyo en toma de decisiones clinicas. "
    "Este modelo no reemplaza el criterio medico profesional."
)
st.divider()

@st.cache_data
def cargar_datos():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi_was_missing'] = df['bmi'].isna().astype(int)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other'].copy()
    df = df[df['age'] >= 18].copy()
    df = df.drop(columns=['id'])
    return df

@st.cache_resource
def entrenar_modelo(df):
    target          = 'stroke'
    num_features    = ['age', 'avg_glucose_level', 'bmi', 'bmi_was_missing']
    binary_features = ['hypertension', 'heart_disease']
    cat_features    = ['gender', 'ever_married', 'work_type',
                       'Residence_type', 'smoking_status']
    all_features    = num_features + binary_features + cat_features

    X = df[all_features]
    y = df[target]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features + binary_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features),
    ])

    modelo = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    modelo.fit(X_train, y_train)
    return modelo, all_features

df     = cargar_datos()
modelo, feature_names = entrenar_modelo(df)

st.subheader("Evaluacion de Riesgo por Paciente")
st.markdown(
    "Complete los datos del paciente para obtener su nivel de riesgo de ACV "
    "y la accion clinica recomendada."
)

col1, col2, col3 = st.columns(3)

with col1:
    edad         = st.slider("Edad (años)", 18, 100, 55)
    glucosa      = st.slider("Glucosa promedio (mg/dL)", 50.0, 300.0, 100.0)
    bmi          = st.slider("IMC (kg/m2)", 10.0, 60.0, 28.0, 0.1)

with col2:
    hipertension = st.selectbox("Hipertension", ["No", "Si"])
    enf_cardiaca = st.selectbox("Enfermedad cardiaca", ["No", "Si"])
    casado       = st.selectbox("Casado/a", ["No", "Yes"])
    genero       = st.selectbox("Genero", ["Male", "Female"])

with col3:
    trabajo      = st.selectbox("Tipo de trabajo",
                       ["Private", "Self-employed", "Govt_job", "children"])
    residencia   = st.selectbox("Residencia", ["Urban", "Rural"])
    tabaco       = st.selectbox("Habito de fumar",
                       ["never smoked", "formerly smoked",
                        "smokes", "Unknown"])

if st.button("Evaluar Riesgo", type="primary", use_container_width=True):
    datos = pd.DataFrame([[
        edad, glucosa, bmi, 0,
        1 if hipertension == "Si" else 0,
        1 if enf_cardiaca == "Si" else 0,
        genero, casado, trabajo, residencia, tabaco
    ]], columns=feature_names)

    prob       = modelo.predict_proba(datos)[0][1]
    prediccion = modelo.predict(datos)[0]

    if prob >= 0.70:
        nivel  = "ALTO"
        accion = "Inscripcion inmediata al programa preventivo"
        color  = "error"
    elif prob >= 0.40:
        nivel  = "MODERADO"
        accion = "Evaluacion adicional en 30 dias"
        color  = "warning"
    else:
        nivel  = "BAJO"
        accion = "Seguimiento anual rutinario"
        color  = "success"

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de ACV", f"{prob*100:.1f}%")
    c2.metric("Nivel de riesgo",     nivel)
    c3.metric("Accion recomendada",  accion)

    if color == "error":
        st.error(
            f"Probabilidad: {prob*100:.1f}% — Riesgo ALTO. "
            "Inscripcion inmediata al programa preventivo. "
            "Solicitar perfil glucemico completo y valoracion cardiologica."
        )
    elif color == "warning":
        st.warning(
            f"Probabilidad: {prob*100:.1f}% — Riesgo MODERADO. "
            "Evaluacion adicional en los proximos 30 dias. "
            "Iniciar consejeria nutricional y control de presion arterial."
        )
    else:
        st.success(
            f"Probabilidad: {prob*100:.1f}% — Riesgo BAJO. "
            "Seguimiento anual rutinario. "
            "Educacion sobre factores de riesgo modificables."
        )

st.divider()
st.markdown(
    "<small>Sistema de apoyo a la decision clinica — Machine Learning | "
    "Este modelo no reemplaza el criterio medico profesional.</small>",
    unsafe_allow_html=True
)
