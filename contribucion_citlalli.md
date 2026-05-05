# Contribución — Citlalli Olmedo

## Sección 1: Planteamiento del Problema
- Definicion del contexto clinico: equipo de atencion primaria con recursos limitados
- Identificacion de la pregunta de negocio: priorizacion de pacientes para programas preventivos
- Vinculacion del modelo con la decision del cliente

## Sección 2: Preparación de Datos
- Carga del dataset Stroke Prediction (5,110 registros, 12 columnas)
- Tratamiento de valores faltantes en columna BMI (imputacion con mediana + flag)
- Eliminacion del registro 'Other' en variable gender
- Restriccion a poblacion adulta (mayores de 18 años)
- Dataset final: 4,253 registros limpios
- Analisis exploratorio (EDA): distribuciones, tasas de ACV por subgrupo, correlaciones
- Seleccion justificada de variables predictoras
