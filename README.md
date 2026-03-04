#  TelecomX LATAM — Predicción de Evasión de Clientes (Churn)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=flat-square&logo=scikit-learn)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11-blueviolet?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-orange?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4c72b0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completado-success?style=flat-square)
![Challenge](https://img.shields.io/badge/Alura-Data%20Science%20Challenge-blue?style=flat-square)

---

##  Descripción del Proyecto

Este proyecto es parte del **Challenge de Data Science — Alura LATAM + Oracle ONE** y está dividido en dos partes:

- **Parte 1 — `TelecomX_LATAM.ipynb`:** ETL (Extracción, Transformación y Carga) + Análisis Exploratorio de Datos (EDA) + Informe Final
- **Parte 2 — `TelecomX_parte2_Latam.ipynb`:** Preparación de datos para ML + Modelos Predictivos + Interpretación y Conclusiones

El objetivo es analizar el comportamiento de **evasión de clientes (Churn)** de la empresa de telecomunicaciones **TelecomX**, identificar los factores que impulsan la cancelación y construir modelos predictivos que permitan anticiparse a la pérdida de clientes.

---

##  Objetivos

- Extraer, limpiar y transformar datos de clientes desde una API JSON
- Realizar análisis exploratorio para identificar patrones de evasión
- Construir y comparar modelos predictivos de Machine Learning
- Interpretar las variables más relevantes para la predicción
- Proponer estrategias de retención basadas en los hallazgos

---

##  Estructura del Repositorio

```
TelecomX_LATAM/
│
├──  TelecomX_LATAM.ipynb            # Parte 1: ETL + EDA + Informe Final
├──  TelecomX_parte2_Latam.ipynb     # Parte 2: ML + Modelos + Conclusiones
├──  TelecomX_Data.json              # Dataset original (fuente: API)
├──  datos_tratados.csv              # Dataset limpio generado en Parte 1 → usado en Parte 2
├──  TelecomX_diccionario.md         # Diccionario de datos
└──  README.md                       # Este archivo
```

---

##  Diccionario de Datos

| Columna | Descripción |
|---|---|
| `ID_Cliente` | ID único del cliente |
| `Evasion` | Variable objetivo: 1 = canceló, 0 = permanece |
| `Adulto_Mayor` | Si el cliente tiene 65 años o más (0/1) |
| `Pareja` | Si tiene pareja (0/1) |
| `Dependientes` | Si tiene dependientes (0/1) |
| `Meses_Contrato` | Antigüedad en meses |
| `Servicio_Internet` | Tipo de servicio: DSL, Fibra Óptica, Sin internet |
| `Tipo_Contrato` | Mes a mes / Un año / Dos años |
| `Metodo_Pago` | Forma de pago del cliente |
| `Cargo_Mensual` | Cargo mensual total ($) |
| `Cargo_Total` | Total acumulado gastado ($) |

>  Ver diccionario completo en [`TelecomX_diccionario.md`](TelecomX_diccionario.md)

---

##  Pipeline Completo del Proyecto

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PARTE 1 — TelecomX_LATAM.ipynb
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXTRACCIÓN          TRANSFORMACIÓN         EDA
─────────────        ─────────────────       ──────────────
API JSON       →     Aplanar JSON      →     Estadísticas
requests             Limpiar nulos           Visualizaciones
json_normalize       Eliminar duplicados     Análisis Churn
                     Encoding                Correlaciones
                     Crear features          Boxplots/Scatter
                            │
                            ▼
                       datos_tratados.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PARTE 2 — TelecomX_parte2_Latam.ipynb
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PREPARACIÓN ML
─────────────────────────────────────────────────
One-Hot Encoding → SMOTE → Split 80/20
         │                      │
         ▼                      ▼
   Regresión              Random Forest
   Logística              (sin escalar)
   (escalado)
         │                      │
         └──────────┬───────────┘
                    ▼
       EVALUACIÓN Y CONCLUSIONES
     Accuracy · Precision · Recall
     F1-Score · ROC-AUC · Confusion Matrix
     Feature Importance · Coeficientes
     Estrategias de retención
```

---

##  Principales Hallazgos — EDA

### Tasa de Evasión General
> Aproximadamente el **26% de los clientes** cancelaron el servicio.

| Factor | Hallazgo | Impacto |
|---|---|---|
| **Tipo de Contrato** | Clientes mes a mes → ~43% de evasión |  Alto |
| **Servicio de Internet** | Fibra óptica presenta mayor tasa de churn |  Alto |
| **Método de Pago** | Cheque electrónico → ~45% de evasión |  Alto |
| **Meses de Contrato** | Los primeros 12 meses son el período crítico |  Medio |
| **Cargo Mensual** | Clientes que evaden pagan más por mes |  Medio |

---

##  Modelos Predictivos

### Modelos entrenados

| Modelo | Normalización | Justificación |
|---|---|---|
| **Regresión Logística** |  StandardScaler | Optimiza coeficientes por gradiente — sensible a escala |
| **Random Forest** |  No requerida | Basado en divisiones por umbrales — invariante a escala |

### Técnicas aplicadas

- **Balanceo de clases:** SMOTE (Synthetic Minority Oversampling Technique)
- **División:** 80% entrenamiento / 20% prueba con estratificación
- **Evaluación:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Matriz de Confusión

>  Los valores de métricas se generan dinámicamente al ejecutar el notebook.

---

##  Variables más Relevantes para la Predicción

###  Variables que aumentan el riesgo de evasión

| Ranking | Variable | Interpretación |
|---|---|---|
| 1 | `Tipo_Contrato_Mes a mes` | Sin compromiso contractual — mayor libertad de cancelar |
| 2 | `Servicio_Internet_Fibra óptica` | Alta insatisfacción pese a ser el servicio premium |
| 3 | `Cargo_Mensual` | Precio alto sin fidelidad genera evasión |
| 4 | `Metodo_Pago_Cheque electrónico` | Menor automatización y compromiso con el servicio |

###  Variables que reducen el riesgo de evasión

| Ranking | Variable | Interpretación |
|---|---|---|
| 1 | `Meses_Contrato` | Mayor antigüedad = mayor retención |
| 2 | `Tipo_Contrato_Dos años` | Compromiso a largo plazo |
| 3 | `Seguridad_Online / Soporte_Tecnico` | Servicios adicionales generan fidelidad |
| 4 | `Metodo_Pago automático` | Correlaciona fuertemente con permanencia |

---

##  Estrategias de Retención Recomendadas

| Prioridad | Estrategia | Segmento objetivo |
|---|---|---|
|  Alta | Incentivar migración a contratos anuales con descuentos | Clientes mes a mes con < 6 meses |
|  Alta | Programa de bienvenida + soporte prioritario | Clientes nuevos (0–3 meses) |
|  Alta | Revisar precios en planes de Fibra Óptica | Clientes fibra + cargo alto |
|  Media | Migrar a pago automático con incentivos (mes gratis) | Clientes con cheque electrónico |
|  Media | Ofrecer servicios adicionales gratuitos por 3 meses | Clientes sin servicios extra |
|  Media | Programa de fidelización con beneficios por antigüedad | Clientes entre 6–18 meses |

### Implementación del Modelo en Producción

1. Generar **score de riesgo mensual** (0–1) para cada cliente activo
2. Activar alertas de retención para clientes con **score > 0.60**
3. Personalizar la oferta según el **factor de riesgo principal** de cada cliente
4. **Reentrenar el modelo** cada trimestre con datos actualizados

---

##  Tecnologías Utilizadas

| Herramienta | Uso |
|---|---|
| `Python 3.10+` | Lenguaje principal |
| `Pandas` | Manipulación y limpieza de datos |
| `NumPy` | Operaciones numéricas |
| `Matplotlib / Seaborn` | Visualizaciones y gráficos |
| `Scikit-Learn` | Modelos ML, métricas, preprocesamiento |
| `Imbalanced-Learn` | Balanceo de clases con SMOTE |
| `Requests` | Extracción de datos desde la API |
| `Jupyter Notebook` | Entorno de desarrollo |

---

##  Cómo Ejecutar el Proyecto

**1. Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/challenge2-data-science-LATAM-parte2
cd challenge2-data-science-LATAM-parte2
```

**2. Instalar dependencias**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn requests jupyter
```

**3. Ejecutar Parte 1**
```bash
jupyter notebook TelecomX_LATAM.ipynb
```
> Al finalizar, exportar el DataFrame limpio como `datos_tratados.csv`

**4. Ejecutar Parte 2**
```bash
jupyter notebook TelecomX_parte2_Latam.ipynb
```

**5. Ejecutar todas las celdas en orden**
> `Kernel` → `Restart & Run All`

---

##  Contenido de los Notebooks

###  TelecomX_LATAM.ipynb — Parte 1

| Sección | Contenido |
|---|---|
|  Extracción | Carga de datos desde la API JSON |
|  Transformación | Limpieza, estandarización y feature engineering |
|  Carga y Análisis | EDA completo: estadísticas, visualizaciones y correlaciones |
|  Informe Final | Conclusiones e insights del análisis exploratorio |

###  TelecomX_parte2_Latam.ipynb — Parte 2

| Sección | Contenido |
|---|---|
|  Preparación de Datos | One-Hot Encoding, SMOTE, split train/test, escalado |
|  Correlación y Variables | Matriz de correlación, boxplots y scatter plots |
|  Modelos Predictivos | Entrenamiento, evaluación y comparación de modelos |
|  Interpretación y Conclusiones | Feature importance, coeficientes y estrategias de retención |

---

##  Juan Camilo Mantilla Ramirez

Hecho con cariño como parte del **Challenge de Data Science — Alura LATAM + Oracle ONE**

---
