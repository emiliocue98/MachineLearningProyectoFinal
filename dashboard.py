import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Configuracion ────────────────────────────────────────────
st.set_page_config(
    page_title="Prediccion de Riesgo de ACV",
    page_icon="🧠",
    layout="wide"
)

st.title("Sistema de Prediccion de Riesgo de ACV")
st.markdown("Dashboard ejecutivo para apoyo en toma de decisiones clinicas")
st.divider()

# ── Cargar y preparar datos ──────────────────────────────────
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
    target = 'stroke'
    X = df.drop(columns=[target])
    y = df[target]

    num_features = ['age', 'avg_glucose_level', 'bmi', 'bmi_was_missing']
    cat_features = ['gender', 'ever_married', 'work_type',
                    'Residence_type', 'smoking_status']
    binary_features = ['hypertension', 'heart_disease']
    all_features = num_features + binary_features + cat_features

    X = X[all_features]

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
    y_prob = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    return modelo, all_features, auc, X_test, y_test

df = cargar_datos()
modelo, feature_names, auc_score, X_test, y_test = entrenar_modelo(df)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Prediccion Individual",
    "Explorar Datos",
    "Metricas del Modelo"
])

# ────────────────────────────────────────────────────────────
# TAB 1 — Prediccion individual
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Evaluacion de Riesgo por Paciente")
    st.markdown("Complete los datos del paciente para obtener su nivel de riesgo de ACV.")

    col1, col2, col3 = st.columns(3)

    with col1:
        edad       = st.slider("Edad (años)", 18, 100, 55)
        glucosa    = st.slider("Glucosa promedio (mg/dL)", 50.0, 300.0, 100.0)
        bmi        = st.slider("IMC (kg/m2)", 10.0, 60.0, 28.0, 0.1)

    with col2:
        hipertension   = st.selectbox("Hipertension", ["No", "Si"])
        enf_cardiaca   = st.selectbox("Enfermedad cardiaca", ["No", "Si"])
        casado         = st.selectbox("Casado/a", ["No", "Yes"])

    with col3:
        genero         = st.selectbox("Genero", ["Male", "Female"])
        trabajo        = st.selectbox("Tipo de trabajo",
                            ["Private", "Self-employed", "Govt_job", "children"])
        residencia     = st.selectbox("Residencia", ["Urban", "Rural"])
        tabaco         = st.selectbox("Habito de fumar",
                            ["never smoked", "formerly smoked",
                             "smokes", "Unknown"])

    if st.button("Evaluar Riesgo", type="primary", use_container_width=True):
        datos = pd.DataFrame([[
            edad, glucosa, bmi, 0,
            1 if hipertension == "Si" else 0,
            1 if enf_cardiaca == "Si" else 0,
            genero, casado, trabajo, residencia, tabaco
        ]], columns=feature_names)

        prob      = modelo.predict_proba(datos)[0][1]
        prediccion = modelo.predict(datos)[0]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de ACV", f"{prob*100:.1f}%")
        c2.metric("Nivel de riesgo", "ALTO" if prediccion == 1 else "BAJO")
        c3.metric("Accion recomendada",
                  "Inscribir en programa preventivo" if prediccion == 1
                  else "Seguimiento rutinario anual")

        if prediccion == 1:
            st.error("El paciente presenta ALTO riesgo de ACV. Se recomienda inscripcion prioritaria en el programa de intervencion preventiva.")
        else:
            st.success("El paciente presenta BAJO riesgo de ACV. Se recomienda seguimiento anual de rutina.")

# ────────────────────────────────────────────────────────────
# TAB 2 — Explorar datos
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploracion del Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pacientes", f"{len(df):,}")
    c2.metric("Con ACV", f"{df['stroke'].sum():,}")
    c3.metric("Sin ACV", f"{(df['stroke']==0).sum():,}")
    c4.metric("Tasa de ACV", f"{df['stroke'].mean()*100:.1f}%")

    st.divider()

    variable = st.selectbox(
        "Seleccionar variable para analizar:",
        ['age', 'avg_glucose_level', 'bmi']
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color, nombre in zip(
            [0, 1], ['#1976D2', '#C62828'], ['Sin ACV', 'Con ACV']
        ):
            ax.hist(df[df['stroke']==label][variable],
                    bins=25, alpha=0.6, color=color, label=nombre)
        ax.set_title(f'Distribucion de {variable}')
        ax.set_xlabel(variable)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [df[df['stroke']==0][variable], df[df['stroke']==1][variable]],
            labels=['Sin ACV', 'Con ACV'],
            patch_artist=True,
            boxprops=dict(facecolor='#E3F2FD'),
            medianprops=dict(color='#1565C0', linewidth=2)
        )
        ax.set_title(f'Boxplot de {variable} por diagnostico')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.divider()
    st.markdown("**Vista previa del dataset:**")
    st.dataframe(df.head(20), use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 3 — Metricas del modelo
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Rendimiento del Modelo — Random Forest")

    st.metric("ROC-AUC en conjunto de prueba", f"{auc_score:.4f}")

    st.divider()

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("**Justificacion del modelo:**")
        st.markdown("""
        Random Forest con class_weight fue seleccionado porque:
        - Maneja el desbalance de clases sin necesidad de sobremuestreo
        - ROC-AUC ~0.81 en conjunto de prueba
        - Captura el 60% de los casos de ACV reales
        - El modelo es 3-4 veces mejor que la tasa base
        - Variables mas importantes: edad, glucosa, hipertension
        """)

    with col_y:
        # Importancia de variables
        rf = modelo.named_steps['classifier']
        ohe = modelo.named_steps['preprocessor']\
              .named_transformers_['cat']
        cat_names = ohe.get_feature_names_out(
            ['gender', 'ever_married', 'work_type',
             'Residence_type', 'smoking_status']
        ).tolist()
        num_bin = ['age', 'avg_glucose_level', 'bmi',
                   'bmi_was_missing', 'hypertension', 'heart_disease']
        all_names = num_bin + cat_names

        importancias = pd.Series(
            rf.feature_importances_, index=all_names
        ).sort_values(ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(6, 4))
        importancias[::-1].plot(kind='barh', ax=ax, color='#1976D2')
        ax.set_title('Top 8 variables mas importantes')
        ax.set_xlabel('Importancia relativa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.divider()
    st.markdown("**Protocolo de decision clinica:**")
    protocolo = pd.DataFrame({
        'Probabilidad predicha': ['Mayor a 70%', 'Entre 40% y 70%', 'Menor a 40%'],
        'Accion recomendada': [
            'Inscripcion inmediata en programa preventivo',
            'Evaluacion adicional en 30 dias',
            'Seguimiento anual rutinario'
        ],
        'Prioridad': ['Alta', 'Media', 'Baja']
    })
    st.dataframe(protocolo, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "<small>Sistema de apoyo a la decision clinica — "
    "Este modelo no reemplaza el criterio medico profesional.</small>",
    unsafe_allow_html=True
)
