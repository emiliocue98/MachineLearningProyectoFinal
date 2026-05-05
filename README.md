# Proyecto Final — Machine Learning
## Predicción de Riesgo de Accidente Cerebrovascular (ACV) para Priorización de Pacientes en Programas de Prevención

Un equipo de atención primaria cuenta con recursos limitados para inscribir pacientes en programas preventivos (monitoreo de presión arterial, consejería nutricional, control glucémico, deshabituación tabáquica). El objetivo de este proyecto es construir un modelo de Machine Learning que, a partir de variables clínicas y de estilo de vida fácilmente disponibles en una consulta general, estime la probabilidad individual de que un paciente sufra un ACV, permitiendo al equipo médico priorizar a quienes presentan mayor riesgo.

**Decisión que habilita el modelo:** ¿A qué pacientes debe el equipo de atención primaria inscribir prioritariamente en su programa de intervención preventiva durante el próximo trimestre?

El reporte se encuentra con el nombre "Reporte_Proyecto_Final.md" en la carpeta github_fix, junto a las imagenes.
---

## Integrantes del equipo
Citlalli Izel Olmedo Paredes
Emilio Cue Funes
Gloria Janeth Esparza Martinez 

---

## Estructura del repositorio
MachineLearningProyectoFinal/
├── ProyectoFinal.ipynb                  Notebook principal de análisis
├── dashboard.py                          Dashboard ejecutivo (Streamlit)
├── healthcare-dataset-stroke-data.csv    Dataset
├── github_fix/
│   └── Reporte_Proyecto_Final.md        Reporte profesional de consultoría
├── contribucion_citlalli.md              Contribución de Citlalli
├── contribucion_emilio.md                Contribución de Emilio
├── contribucion_janeth.md                Contribución de Janeth
└── README.md                             Este archivo

---

## Instalación y ejecución

```bash
# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn streamlit

# Ejecutar el notebook
jupyter notebook ProyectoFinal.ipynb

# Lanzar el dashboard
streamlit run dashboard.py
```

> **Nota:** Para ejecutar el dashboard y el notebook es necesario tener el archivo `healthcare-dataset-stroke-data.csv` en la misma carpeta del proyecto.
```

---

## Resultados principales

| Modelo | ROC-AUC | Recall | F1-Score |
|--------|---------|--------|----------|
| Regresión Logística + SMOTE | 0.81 | ~60% | — |
| Random Forest + class_weight | 0.81 | ~60% | — |

**Modelo seleccionado:** Regresión Logística — mejor generalización (gap train-test ~2.5% vs ~9% de Random Forest).

**Hallazgo clave:** el modelo identifica casos de ACV 3-4 veces mejor que la tasa base, con las variables edad, glucosa, hipertensión y enfermedad cardiaca como los predictores más importantes.

---

## Tecnologías utilizadas

- Python 3.11
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas / numpy
- matplotlib / seaborn
- Streamlit