#  Glassdoor Sentiment Analysis — Challenge Basico

[![Status](https://img.shields.io/badge/Status-Completado-brightgreen)](https://github.com/HectorDelgado9997/challenge-basico)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![VADER](https://img.shields.io/badge/VADER-Lexical-yellow)](https://github.com/cjhutto/vaderSentiment)
[![pysentimiento](https://img.shields.io/badge/pysentimiento-Transformer-purple)](https://github.com/pysentimiento/pysentimiento)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![CI](https://github.com/HectorDelgado9997/challenge-basico/actions/workflows/ci.yml/badge.svg)](https://github.com/HectorDelgado9997/challenge-basico/actions/workflows/ci.yml)
##  Descripcion

Pipeline de analisis de sentimiento sobre reviews de empleados de Glassdoor.
El sistema compara dos enfoques complementarios:

- **VADER** — modelo lexical basado en reglas, rapido y sin necesidad de GPU
- **pysentimiento** — modelo transformer basado en RoBERTa multilingue

Ambos modelos clasifican cada review en **positive**, **neutral** o **negative**,
y el pipeline mide el nivel de acuerdo entre ambos enfoques.

---

##  Objetivo

Analizar el sentimiento de reviews de Glassdoor usando dos metodologias
distintas, comparar sus resultados, y registrar todo el experimento
automaticamente con MLflow.

---

##  Estructura del Repositorio

```text
challenge-basico/
├── data/
│   ├── raw/
│   │   └── glassdoor_comments.csv          ← Dataset fuente
│   └── processed/
│       └── glassdoor_sentiment_results.csv ← Resultados con labels
├── docs/
│   ├── architecture.md                     ← Arquitectura y flujo de datos
│   ├── dataset_extraction.md               ← Dataset y proceso de carga
│   ├── model_construction.md               ← Modelos y decisiones de diseno
│   ├── mlops.md                            ← Configuracion MLflow
│   ├── results.md                          ← Resultados del experimento
│   └── technical_run_guide.md              ← Guia de ejecucion paso a paso
├── outputs/
│   ├── figures/
│   │   └── sentiment_distribution.png      ← Grafica comparativa
│   └── reports/
│       └── model_comparison_report.txt     ← Reporte de acuerdo y metricas
├── src/
│   ├── ingestion.py                        ← Carga y validacion del CSV
│   ├── preprocessing.py                    ← Normalizacion y tokenizacion
│   ├── sentiment_vader.py                  ← Scoring VADER
│   ├── sentiment_pysentimiento.py          ← Scoring pysentimiento
│   ├── run_sentiment_analysis.py           ← Pipeline sin MLflow
│   ├── evaluation.py                       ← Metricas, graficas y reporte
│   └── mlflow_pipeline.py                 ← Pipeline completo con MLflow
├── params.yaml                             ← Configuracion centralizada
├── requirements.txt
└── README.md
```

---

##  Modelos

| Modelo | Tipo | Descripcion |
|---|---|---|
| VADER | Lexical / Reglas | Compound score → positive / neutral / negative |
| pysentimiento | Transformer (RoBERTa) | Modelo multilingue para NLP social |

### Thresholds VADER

| Compound score | Label |
|---|---|
| >= 0.05 | positive |
| <= -0.05 | negative |
| entre -0.05 y 0.05 | neutral |

---

##  Preprocesamiento

El pipeline aplica **dos normalizaciones distintas** sobre el mismo texto:

| Normalizacion | Uso | Preserva |
|---|---|---|
| `normalize_for_sentiment` | VADER + pysentimiento | Negaciones, intensificadores, puntuacion |
| `normalize_for_features` | TF-IDF, n-gramas | Solo texto limpio lowercase |

---

##  Dataset

| Propiedad | Valor |
|---|---|
| Fuente | Reviews de empleados en Glassdoor |
| Columnas requeridas | `headline`, `pros`, `cons` |
| Columna construida | `review_text` = headline + pros + cons |
| Formato | CSV |
| Encodings soportados | utf-8, utf-8-sig, latin1, ISO-8859-1, cp1252 |

---

##  Instalacion y Ejecucion Rapida

```bash
# 1. Clonar el repositorio
git clone https://github.com/HectorDelgado9997/challenge-basico.git
cd challenge-basico

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate       # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar MLflow (terminal 1)
mlflow ui --host 127.0.0.1 --port 5000

# 5. Ejecutar el pipeline completo (terminal 2)
cd src
python mlflow_pipeline.py
```

> Para instrucciones detalladas ver [`docs/technical_run_guide.md`](docs/technical_run_guide.md)

---

##  Pipeline Completoglassdoor_comments.csv
│
▼
Ingestion → validacion + build review_text
│
▼
Preprocessing → sentiment_text + processed_text
│
├── VADER sentiment scoring
└── pysentimiento sentiment scoring
│
▼
Evaluation → agreement_rate + distribution + crosstab
│
├── outputs/reports/model_comparison_report.txt
├── outputs/figures/sentiment_distribution.png
└── data/processed/glassdoor_sentiment_results.csv
│
▼
MLflow → params + metrics + artifacts registrados

---

##  Metricas Registradas en MLflow

| Metrica | Descripcion |
|---|---|
| `rows_after_ingestion` | Filas validas tras carga y limpieza |
| `rows_after_preprocessing` | Filas validas tras preprocesamiento |
| `agreement_rate` | Proporcion de reviews donde VADER y pysentimiento coinciden |
| `vader_positive_share` | Proporcion de reviews clasificadas como positive por VADER |
| `vader_neutral_share` | Proporcion neutral por VADER |
| `vader_negative_share` | Proporcion negative por VADER |
| `pysentimiento_positive_share` | Proporcion positive por pysentimiento |
| `pysentimiento_neutral_share` | Proporcion neutral por pysentimiento |
| `pysentimiento_negative_share` | Proporcion negative por pysentimiento |

---

##  Tests

```bash
pytest -v
```

---

##  Documentacion

| Archivo | Contenido |
|---|---|
| `docs/dataset_extraction.md` | Dataset, columnas, validaciones y flujo de carga |
| `docs/architecture.md` | Arquitectura, modulos y flujo completo de datos |
| `docs/model_construction.md` | Modelos, normalizacion y decisiones de diseno |
| `docs/mlops.md` | Configuracion MLflow y metricas registradas |
| `docs/results.md` | Resultados del experimento |
| `docs/technical_run_guide.md` | Guia paso a paso de instalacion y ejecucion |

---

##  Stack Tecnologico

| Herramienta | Uso |
|---|---|
| Python 3.9+ | Lenguaje principal |
| pandas / numpy | Manipulacion de datos |
| vaderSentiment | Modelo lexical de sentimiento |
| pysentimiento | Modelo transformer de sentimiento |
| transformers + torch | Backend HuggingFace |
| matplotlib | Visualizaciones |
| mlflow | Tracking de experimentos |
| PyYAML | Lectura de params.yaml |
| langdetect | Deteccion de idioma |
| pytest | Pruebas automatizadas |

---

##  Autor

**Héctor Manuel Delgado Zambrano**
[![GitHub](https://img.shields.io/badge/GitHub-HectorDelgado9997-black)](https://github.com/HectorDelgado9997)
