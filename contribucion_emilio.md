# Contribución — Emilio Cue

## Sección 3: Modelado
- Pipeline de preprocesamiento con ColumnTransformer
- Estandarizacion de variables numericas (StandardScaler)
- One-Hot Encoding para variables categoricas
- Modelo 1: Regresion Logistica con SMOTE para manejo de desbalance de clases
- Modelo 2: Random Forest con class_weight para manejo de desbalance
- Justificacion tecnica de cada algoritmo

## Sección 4: Evaluación
- Validacion cruzada estratificada de 5 pliegues
- Metricas seleccionadas: ROC-AUC, Average Precision, Recall, F1
- Matrices de confusion para ambos modelos
- Curvas ROC y Precision-Recall comparativas
- Diagnostico de overfitting: comparacion train vs test
- Logistic Regression: gap AUC ~2.5% (sin sobreajuste)
- Random Forest: gap AUC ~9% (leve sobreajuste)
- Resultado: AUC final ~0.81
