<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (15)" src="https://github.com/user-attachments/assets/39ffcf62-f25e-49b6-81b7-297bc0604a4f" />

# 🎬 Π Proyecto Pi | Clasificación de Sentimientos en Reseñas de IMDb (Film Junky Union)

## 📋 Descripción
La plataforma **Film Junky Union** busca desarrollar un sistema automatizado capaz de detectar reseñas negativas de películas. Utilizando un conjunto de datos histórico de **IMDb** con 47.331 reseñas etiquetadas, este proyecto implementa y compara múltiples modelos de Procesamiento de Lenguaje Natural (PNL) para clasificar la polaridad del sentimiento de cada opinión.

El criterio principal de éxito exigido por el negocio es alcanzar un puntaje **F1 Score ≥ 0.85** sobre datos no vistos.

## 🎯 Objetivos
- ✅ Realizar un análisis exploratorio (EDA) para comprender la distribución temporal de películas, la densidad de reseñas por título y el balance de clases.
- ✅ Aplicar técnicas de normalización y limpieza textual (remoción de signos/dígitos, filtrado de *stopwords* con **NLTK** y lematización con **spaCy**).
- ✅ Vectorizar los documentos empleando **TF-IDF** (`TfidfVectorizer`).
- ✅ Entrenar, evaluar y comparar distintos clasificadores (**DummyClassifier**, **Regresión Logística** y **LightGBM**) asegurando superar el umbral de **F1 ≥ 0.85**.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn, NLTK, spaCy, LightGBM, Matplotlib y Seaborn
* **Entorno:** Jupyter Notebook (`Proyecto_Pi_Clasificación_de_Sentimientos_IMDb.ipynb`)
* **Técnicas:** Procesamiento de Lenguaje Natural (PNL), Normalización Textual, Lematización, Vectorización TF-IDF, Evaluación con Curvas ROC/PRC y Clasificación Binaria.

## 📊 Fuente de Datos y Estructura
El conjunto de datos (`imdb_reviews.tsv`) consta de **47.331 reseñas** en inglés con 17 variables:
* **`review`**: Texto completo de la crítica cinematográfica.
* **`pos`**: Variable objetivo binaria (`1` = Reseña positiva, `0` = Reseña negativa).
* **`ds_part`**: Indicador de partición predefinida (`23.796` registros para entrenamiento y `23.535` para prueba).
* **Balance de clases:** 23.715 reseñas negativas frente a 23.616 positivas (diferencia de apenas 99 registros), constituyendo un dataset naturalmente balanceado.

## ⚙️ Metodología
1. **Análisis Exploratorio (EDA):** Verificación del balanceo de clases y confirmación de distribuciones equivalentes de puntuaciones y polaridades entre entrenamiento y prueba.
2. **Preprocesamiento y Normalización:** Limpieza mediante expresiones regulares (`clear_text`), filtrado de palabras vacías en inglés (**NLTK**) y lematización (**spaCy** `en_core_web_sm`).
3. **Vectorización:** Transformación del corpus textual a matrices numéricas dispersas mediante **TF-IDF**.
4. **Entrenamiento y Evaluación:** Comparación sistemática empleando Exactitud (*Accuracy*), **F1 Score**, **ROC-AUC** y Puntuación de Precisión Media (**APS**), complementado con gráficos de umbral F1, curvas ROC y curvas Precision-Recall.

## 💡 Comparación de Resultados en Prueba
| Modelo Evaluado | Exactitud | F1 Score | APS | ROC-AUC | Criterio (F1 ≥ 0.85) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modelo 0: DummyClassifier (Constante)** | 0.50 | 0.00 | 0.50 | 0.50 | ❌ |
| **Modelo 1: NLTK + TF-IDF + Regresión Logística** | **0.88** | **0.88** | **0.95** | **0.95** | ✅ |
| **Modelo 2: spaCy (Lematización) + TF-IDF + Reg. Logística** | **0.88** | **0.88** | **0.95** | **0.95** | ✅ |
| **Modelo 3: spaCy + TF-IDF + LGBMClassifier** | 0.86 | 0.86 | 0.93 | 0.94 | ✅ |

### 🏆 Modelo Recomendado: `NLTK + TF-IDF + Regresión Logística`
Con un **F1 Score de 0.88** y un **ROC-AUC de 0.95** en el conjunto de prueba, la combinación de **TF-IDF con Regresión Logística** ofrece el equilibrio ideal entre precisión predictiva, interpretabilidad y velocidad de entrenamiento/inferencia frente a enfoques más pesados como LightGBM o lematización exhaustiva.

## 🧪 Validación con Reseñas Personalizadas
| Reseña de Prueba (Inglés) | Prob. Positiva (Modelo 1) | Interpretación |
| :--- | :---: | :--- |
| *"I did not simply like it, not my kind of movie."* | **0.14** | Negativa ❌ |
| *"Well, I was bored and felt asleep in the middle of the movie."* | **0.16** | Negativa ❌ |
| *"I was really fascinated with the movie"* | **0.54** | Positiva (Neutra/Leve) ✅ |
| *"Even the actors looked really old and disinterested..."* | **0.11** | Negativa ❌ |
| *"The movie had its upsides and downsides, but I feel like overall it's a decent flick..."* | **0.47** | Mixta / Ambigua ⚖️ |
| *"What a rotten attempt at a comedy. Not a single joke lands..."* | **0.04** | Muy Negativa ❌ |
| *"Launching on Netflix was a brave move & I really appreciate..."* | **0.82** | Muy Positiva ✅ |

## 📈 Principales Hallazgos
* **Eficacia de TF-IDF + Modelos Lineales:** La Regresión Logística superó a LightGBM (`F1 = 0.88` vs `0.86`) con un costo computacional significativamente menor.
* **Impacto de la lematización:** Aplicar lematización con spaCy mantuvo prácticamente idéntico el desempeño en prueba (`F1 = 0.88`, `ROC-AUC = 0.95`) respecto al flujo directo con NLTK y TF-IDF.
* **Sensibilidad tonal:** En las pruebas con reseñas personalizadas, los modelos lineales captaron adecuadamente tanto críticas contundentes como matices mixtos.

## 📂 Archivos Principales
* `Proyecto_Pi_Clasificación_de_Sentimientos_IMDb.ipynb`
* `imdb_reviews.tsv`

## 🏁 Conclusión
Se desarrolló con éxito un clasificador automático de sentimientos para **Film Junky Union** basado en **NLTK, TF-IDF y Regresión Logística**, alcanzando un **F1 Score de 0.88** y un **ROC-AUC de 0.95** sobre datos no vistos. La solución supera el requisito técnico (`F1 ≥ 0.85`) y brinda una herramienta rápida, escalable y económica para filtrar reseñas negativas en la plataforma.de opiniones, moderación de contenido y análisis de satisfacción de usuarios.
