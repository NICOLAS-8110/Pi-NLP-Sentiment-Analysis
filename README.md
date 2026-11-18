<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (15)" src="https://github.com/user-attachments/assets/39ffcf62-f25e-49b6-81b7-297bc0604a4f" />

# 🎬 Π Proyecto Pi | Clasificación de Sentimientos en Reseñas de IMDb (Film Junky Union)

## 📋 Descripción
La plataforma **Film Junky Union** busca desarrollar un sistema automatizado capaz de detectar reseñas negativas de películas. Utilizando un conjunto de datos histórico de **IMDb** con más de 47.000 reseñas etiquetadas, este proyecto implementa y compara múltiples modelos de Procesamiento de Lenguaje Natural (PNL) para clasificar la polaridad del sentimiento de cada opinión.

El criterio principal de éxito exigido por el negocio es alcanzar un puntaje **F1 Score ≥ 0.85** sobre datos no vistos.

## 🎯 Objetivos
- ✅ Realizar un análisis exploratorio (EDA) para comprender la distribución temporal de películas, la densidad de reseñas por título y el balance de clases.
- ✅ Aplicar técnicas de normalización y limpieza textual (minúsculas, remoción de signos, números y *stopwords* con **NLTK**, además de lematización con **spaCy**).
- ✅ Vectorizar los documentos empleando **TF-IDF** y explorar representaciones contextuales con **BERT** (`bert-base-uncased`).
- ✅ Entrenar, validar y comparar distintos clasificadores (Regresión Logística, LightGBM y BERT) asegurando superar el umbral de **F1 ≥ 0.85**.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn, NLTK, spaCy, LightGBM, PyTorch, Transformers (Hugging Face), Matplotlib y Seaborn
* **Entorno:** Jupyter Notebook (`Proyecto_Pi_Clasificacion_de_Sentimientos_IMDb.ipynb`)
* **Técnicas:** Procesamiento de Lenguaje Natural (PNL), Normalización Textual, Lematización, Vectorización TF-IDF, Embeddings con BERT y Clasificación Binaria.

## 📊 Fuente de Datos y Estructura
El conjunto de datos (`imdb_reviews`) consta de **47.331 reseñas** en inglés con 17 variables:
* **`review`**: Texto completo de la crítica cinematográfica.
* **`pos`**: Variable objetivo binaria (`1` = Reseña positiva, `0` = Reseña negativa).
* **Balance de clases:** 23.715 reseñas negativas frente a 23.616 positivas (diferencia de apenas 99 registros), constituyendo un dataset naturalmente balanceado.

## ⚙️ Metodología
1. **Análisis Exploratorio (EDA):** Verificación del balanceo de clases y confirmación de particiones representativas entre los conjuntos de entrenamiento y prueba.
2. **Preprocesamiento y Normalización:** Estandarización del texto mediante conversión a minúsculas, eliminación de caracteres especiales/números, filtrado de palabras vacías (**NLTK**) y lematización (**spaCy**).
3. **Vectorización y Embeddings:** Transformación del corpus mediante **TF-IDF** y extracción de vectores `[CLS]` utilizando el modelo preentrenado `bert-base-uncased`.
4. **Entrenamiento y Evaluación:** Comparación de desempeño empleando Precisión, **F1 Score**, **ROC-AUC** y Puntuación de Precisión Media (**APS**), complementado con curvas ROC y Precision-Recall.

## 💡 Comparación de Resultados en Prueba
| Modelo Evaluado | Precisión | F1 Score | ROC-AUC | APS | Criterio (F1 ≥ 0.85) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier (Línea Base)** | 0.50 | 0.50 | 0.50 | 0.50 | ❌ |
| **TF-IDF + Regresión Logística (Seleccionado)** | **0.88** | **0.88** | **0.95** | **0.95** | ✅ |
| **spaCy + Lematización + TF-IDF + Reg. Logística** | 0.87 | 0.87 | 0.95 | 0.94 | ✅ |
| **spaCy + TF-IDF + LightGBM** | 0.85 | 0.85 | 0.93 | 0.93 | ✅ |

*(Nota técnica: También se implementó un pipeline con **BERT + Regresión Logística**; sin embargo, ante la ausencia de GPU en el entorno de ejecución, se evaluó sobre una muestra reducida).*

### 🏆 Mejor Modelo: `TF-IDF + Regresión Logística`
Con un **F1 Score de 0.88** y un **ROC-AUC de 0.95** en el conjunto de prueba, la combinación de **TF-IDF con Regresión Logística** ofreció el mejor rendimiento global, superando la meta de negocio con baja complejidad computacional y alta velocidad de inferencia.

## 🧪 Validación con Reseñas Personalizadas
| Reseña de Prueba | Probabilidad Positiva | Predicción |
| :--- | :---: | :---: |
| *"Esta película fue absolutamente increíble y las actuaciones fueron fantásticas."* | **0.95** | Positiva ✅ |
| *"Excelente película. Sin duda la volvería a ver."* | **1.00** | Positiva ✅ |
| *"Argumento pésimo y personajes aburridos."* | **0.00** | Negativa ❌ |
| *"Una de las peores películas que he visto en mi vida."* | **0.46** | Negativa ❌ |
| *"La historia era interesante, pero el ritmo era lento."* | **0.19** | Negativa ❌ |

## 📈 Principales Hallazgos
* **Eficacia de modelos lineales sobre texto disperso:** La Regresión Logística combinada con TF-IDF superó tanto a algoritmos basados en árboles (LightGBM) como a pipelines con lematización pesada, manteniendo intactos matices relevantes del vocabulario.
* **Escalabilidad operativa:** El modelo seleccionado permite procesar y clasificar miles de reseñas en segundos sin requerir infraestructura GPU costosa en producción.
* **Comportamiento ante ambigüedad:** El sistema distingue con altísima confianza opiniones marcadamente positivas o negativas, reflejando probabilidades intermedias en críticas mixtas.

## 📂 Archivos Principales
* `Proyecto_Pi_Clasificacion_de_Sentimientos_IMDb.ipynb`
* `imdb_reviews.tsv` (o `imdb_reviews.7z`)

## 🏁 Conclusión
Se desarrolló con éxito un clasificador automático de sentimientos para **Film Junky Union** basado en **TF-IDF y Regresión Logística**, alcanzando un **F1 Score de 0.88** y un **ROC-AUC de 0.95** sobre datos no vistos. La solución supera ampliamente el requisito técnico (`F1 ≥ 0.85`) y proporciona una herramienta escalable para el monitoreo continuo de opiniones, moderación de contenido y análisis de satisfacción de usuarios.
