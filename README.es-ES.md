# [ICML 2026] ¡Es TIME: Hacia la Próxima Generación de Conjuntos de Prueba para la Predicción de Series Temporales


[![arXiv](https://img.shields.io/badge/arxiv-2602.12147-b31b1b.svg)](https://arxiv.org/abs/2602.12147)  
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME/tree/main)  
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LeaderBoard-FFD21E)](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)  
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CSVFiles-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)  
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Results&Features-FFD21E)](https://huggingface.co/datasets/Real-TSF/TIME-Output)  
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

TIME es un conjunto de pruebas centrado en tareas para la predicción de series temporales, compuesto por diversos conjuntos de datos novedosos, diseñado específicamente para la evaluación de modelos de predicción de series temporales (TSFM) sin necesidad de entrenamiento previo (zero-shot). Este código proporciona un flujo de trabajo completo desde la preprocesamiento de datos hasta la evaluación del modelo.

## 📅 Registro de Actualizaciones

### 2026 Junio  
- Publicamos la versión final lista para publicación en [arxiv](https://arxiv.org/abs/2602.12147) y [OpenReview](https://openreview.net/forum?id=79TgfXHbsK).  
- Agregamos MSTL como opción para el cálculo de características temporales (STL sigue siendo el valor por defecto, como se usó en el artículo).

### 2026 Mayo  
- 🎉 ¡Nuestro artículo fue aceptado en ICML 2026!  
- Actualización de los conjuntos de datos y la tabla de clasificación:  
   - Actualizamos la licencia del conjunto de datos a CC BY-NC 4.0 para cumplir con los requisitos de todos los proveedores de datos constituentes.  
   - Actualización de Crypto/D. Corregimos los datos con agregados de mercado público.  
   - Actualización de Global_Influenza/W. Solucionamos el error reportado en [Issue #5](https://github.com/zqiao11/TIME/issues/5).

### 2026 Febrero  
- Lanzamiento oficial de nuestro código TIME. Características limpias y ProcessedCSV disponibles en HuggingFace.  
- Actualización de resultados en la tabla de clasificación:  
  - **Chronos2 & Chronos-bolt**: Integramos las actualizaciones de [PR#2](https://github.com/zqiao11/TIME/pull/2).  
  - **TiRex**: Integramos las actualizaciones de [PR#3](https://github.com/zqiao11/TIME/pull/3).  
- Primera publicación de nuestro [artículo en arxiv](https://arxiv.org/abs/2602.12147) y [tabla de clasificación](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard).

## ⚙️ Instalación

1. Recomendamos usar Conda para gestionar el entorno

```bash
conda create -n timebench python=3.11 -y
conda activate timebench
pip install -e .
```

2. Descargue el conjunto de datos desde [huggingface](https://huggingface.co/datasets/Real-TSF/TIME)

3. Defina la ruta para los conjuntos de datos de HF en `.env`. (Usado como `storage_env_var` en [`Dataset`](src/timebench/evaluation/data.py#L120)).

```bash
echo "TIME_DATASET=PATH_TO_DATASET" >> .env
```

## 🚀 Primeros Pasos

### Predicción del Modelo  
Proporcionamos el código completo y los scripts necesarios para reproducir todos los resultados de nuestro conjunto de pruebas.

Para cada modelo, utilice el script correspondiente en el directorio `scripts/` para configurar automáticamente el entorno de Conda y ejecutar evaluaciones en todas las tareas.

⚠️ **Nota Importante**: Asegúrese de que el nombre del entorno de Conda del script no entre en conflicto con otros existentes.

```
# Ejemplo: Ejecutar la evaluación para Chronos2
bash scripts/run_chronos2.sh

# Recomendamos usar nohup para ejecutar los scripts en segundo plano
nohup bash scripts/run_chronos2.sh > run_chronos2.txt 2>&1 &
```

Para cada tarea, las predicciones a nivel de ventana (cuantiles) y métricas se guardarán en `output/results/{model_name}/{dataset}/{freq}/{term}/`.

### Calcular Métricas Generales  
Una vez completadas las evaluaciones, utilice el siguiente script para agrupar las salidas crudas en métricas generales para la tabla de clasificación. Este proceso recupera automáticamente los resultados de Seasonal Naive desde Hugging Face y calcula las métricas agregadas en todas las tareas.

```bash
# Calcular tabla de clasificación general basada en `output/results` (ordenado por MASE)
python scripts/compute_local_leaderboard.py
```

Para un análisis más profundo, incluyendo desglose por conjunto de datos, evaluación por niveles de patrón y visualizaciones, puede descargar y ejecutar localmente nuestra [Aplicación de Tabla de Clasificación](https://huggingface.co/spaces/Real-TSF/TIME-Leaderboard).

## 💻 Ejecutar su Propio Modelo  

Para agregar un nuevo modelo, siga estos pasos:

1. **Implemente su modelo en `experiments/`**

   Cree un nuevo script en Python en el directorio `experiments/` (por ejemplo, `experiments/your_model.py`). Puede usar implementaciones existentes como `experiments/chronos2.py` como plantilla de referencia.

- **Use la clase Dataset**

   La clase `Dataset` se adapta de [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py) y proporciona una interfaz unificada para cargar datos de series temporales:

   ```python
   from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config

   # ⚠️ Importante: Establezca to_univariate según las capacidades de su modelo
   # Si su modelo solo admite predicción univariada:
   to_univariate = False if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1 else True

   # Si su modelo admite predicción multivariada de forma nativa:
   to_univariate = False

   dataset = Dataset(
       name=dataset_name,
       term=term,  # "corto", "mediano" o "largo"
       to_univariate=to_univariate,
       prediction_length=prediction_length,
       test_length=test_length,
       val_length=val_length,
   )
   ```

- **Genere predicciones y guarde resultados**

   TIME utiliza una interfaz de evaluación flexible que no depende de GluonTS. Simplemente calcule las predicciones de cuantiles (`fc_quantiles`) externamente y páselas a `save_window_predictions`:

   ```python
   from timebench.evaluation.saver import save_window_predictions

   # Genere fc_quantiles con forma:
   # - (num_total_instances, num_quantiles, prediction_length) para univariado
   # - (num_total_instances, num_quantiles, num_variates, prediction_length) para multivariado
   # donde num_total_instances = num_series_exp * num_windows

   save_window_predictions(
       dataset=dataset,
       fc_quantiles=fc_quantiles,
       ds_config=f"{dataset_name}/{freq}/{term}",
       output_base_dir="output/results",
       seasonality=season_length,
       model_hyperparams={"model_name": "your_model"},
   )
   ```

   Esta función calcula automáticamente las métricas por ventana y guarda predicciones, métricas y archivos de configuración en `output/results/{model_name}/{dataset}/{freq}/{term}/`.

2. **Cree un script de ejecución en `scripts/`**

   Cree un script en shell (por ejemplo, `scripts/run_your_model.sh`) para ejecutar su modelo en todas las tareas. El script debe:
   - Configurar el entorno de Conda con las dependencias requeridas  
   - Llamar a su script de experimentos para cada tarea  
   - Incluir configuración específica de hipers parámetros y asegurar reproducibilidad

### Enviar Resultados a la Tabla de Clasificación de TIME  

Una vez completada la evaluación y listo para aparecer en la tabla de clasificación de TIME:
- Abra un Pull Request para subir su carpeta `output/results/{model_name}/` al [repositorio TIME-Output](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/results) en Hugging Face.
      ```python
      from huggingface_hub import HfApi

      api = HfApi()

      model_name = "YOUR_MODEL_NAME"

      api.upload_folder(
         folder_path=f"output/results/{model_name}",  # Ruta a su carpeta local de resultados
         path_in_repo=f"results/{model_name}",
         repo_id="Real-TSF/TIME-Output",
         repo_type="dataset",
         commit_message=f"Submit evaluation results for {model_name}",
         create_pr=True
      )
      ```
- Los resultados se incluirán automáticamente en la tabla de clasificación después de la revisión  
- Para garantizar la reproducibilidad, le recomendamos encarecidamente contribuir su código de experimentos y scripts de ejecución a este repositorio de GitHub.

## 📊 Conjuntos de Datos y Características Temporales

Nuestro código proporciona utilidades para el preprocesamiento de datos y el cálculo de características temporales. Para instrucciones detalladas, consulte la documentación en el directorio `docs/`:
- [Guía de Preprocesamiento de Datos](docs/PREPROCESS.md): Pantallado, preprocesamiento y limpieza de conjuntos de datos en formato CSV  
- [Especificación de Formato de Datos](docs/DATA_FORMAT.md): Conversión de archivos CSV procesados al formato eficiente Arrow  
- [Características de Series Temporales](docs/FEATURES.md): Cálculo de características temporales de archivos CSV procesados  

### Agregar Nuevos Conjuntos de Datos

Si desea agregar un nuevo conjunto de datos a TIME:

1. **Preprocese sus datos** siguiendo la documentación en `docs/`:  
   - Genere archivos CSV procesados  
   - Cree conjuntos de datos Arrow (hf_dataset)  
   - Calcule características temporales

2. **Suba datos procesados a HuggingFace mediante PR**:  
   - Suba archivos CSV procesados a [TIME-ProcessedCSV](https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV)  
   - Suba hf_dataset a [TIME](https://huggingface.co/datasets/Real-TSF/TIME)  
   - Suba características a [TIME-Output](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/features)

3. **Actualice la configuración**:  
   - Actualice `src/timebench/config/datasets.yaml` en GitHub para incluir sus tareas de predicción  
   - Abra un Pull Request con sus cambios

4. **Revisión e integración**:  

   Después de la revisión y aprobación, haremos:  
     - Agregará su conjunto de datos a TIME  
     - Evaluará modelos existentes en sus nuevos conjuntos de datos  
     - Actualizará la tabla de clasificación con nuevos resultados

## 🤝 Agradecimientos

Los componentes principales de este repositorio incluyen código adaptado de los siguientes proyectos destacados:
* [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval)  
* [biblioteca tsfeatures](https://github.com/Nixtla/tsfeatures)

También extendemos nuestro más sincero agradecimiento a los autores de los TSFMs evaluados por compartir su trabajo y promover el progreso en la comunidad de series temporales.

## Cita

Si encuentra este conjunto de pruebas útil, considere citar:
```
@article{qiao2026s,
  title={It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks},
  author={Qiao, Zhongzheng and Pan, Sheng and Wang, Anni and Zhukova, Viktoriya and Liu, Yong and Jiang, Xudong and Wen, Qingsong and Long, Mingsheng and Jin, Ming and Liu, Chenghao},
  journal={arXiv preprint arXiv:2602.12147},
  year={2026}
}
```
