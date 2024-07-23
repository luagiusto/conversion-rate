from scipy.stats._distn_infrastructure import rv_frozen
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
import gzip
import os
import unidecode


def load_comp_data(path_to_data, sample_frac=1.0):
    """
    Carga y preprocesa los datos de competencia de los archivos de entrenamiento y prueba.
    
    Args:
        path_to_data (str): Ruta al directorio que contiene los archivos de datos.
        sample_frac (float, optional): Fracción de datos de entrenamiento a muestrear. Por defecto es 1.0.
        
    Returns:
        pd.DataFrame: Datos de competencia preprocesados con filas de entrenamiento y evaluación.
    """

    # Construir rutas de archivos utilizando os.path.join()
    train_file = os.path.join(path_to_data, "train.csv.gz")
    eval_file = os.path.join(path_to_data, "test.csv")

    # Cargar datos de entrenamiento desde el archivo CSV comprimido con gzip
    with gzip.open(train_file, "rt", encoding="utf-8") as f:
        train_data = pd.read_csv(f)

    # Cargar datos de evaluación desde el archivo CSV
    eval_data = pd.read_csv(eval_file)

    # Realizar muestreo estratificado si sample_frac es menor que 1.0
    if sample_frac < 1.0:
        train_data, _ = train_test_split(
            train_data,
            train_size=sample_frac,
            stratify=train_data["conversion"]
        )

    # Agregar columna "train_eval" para indicar filas de entrenamiento y evaluación
    train_data["train_eval"] = "train"
    eval_data["train_eval"] = "eval"

    # Concatenar verticalmente los datos de entrenamiento y evaluación
    df = pd.concat([train_data, eval_data], axis=0, ignore_index=True)

    return df


def random_search(param_dist, dtrain, watchlist, n_iter=50):
    print(f"Running Random Search with {n_iter} iterations")
    exp_results = []
    best_score = -np.inf
    best_params = None

    for i in range(n_iter):
        params = {}
        for k, v in param_dist.items():
            if isinstance(v, rv_frozen):
                params[k] = v.rvs()
            else:
                params[k] = np.random.choice(v)
        num_boost_round = params.pop("num_boost_round")

        evals_result = {}
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=watchlist, evals_result=evals_result,
                          verbose_eval=False)
        train_auc = evals_result['train']["auc"][-1]
        val_auc = evals_result['validation']["auc"][-1]

        if val_auc > best_score:
            best_score = val_auc
            best_params = params
            best_params["num_boost_round"] = num_boost_round
            best_model = model

        params.update({"train_auc": train_auc, "val_auc": val_auc, "num_boost_round": num_boost_round})
        exp_results.append(params)
        print(f"Iteration {i + 1}/{n_iter} - AUC: {val_auc:.4f} - Params: {params}")

    exp_results = pd.DataFrame(exp_results)
    exp_results = exp_results.sort_values(by="val_auc", ascending=False)
    return best_params, best_score, best_model, exp_results


def to_ralas(df):
    # Convertir los conjuntos de datos a matrices ralas en formato CSR
    df = df.astype(pd.SparseDtype(float, fill_value=0))
    df_names = df.columns
    df = df.sparse.to_coo().tocsr()
    return df, df_names


def process_text(text, rm_s=True):
    # Convertir a minúsculas, eliminar tildes y remover 's' al final
    text = text.lower()  # Convertir a minúsculas
    text = unidecode.unidecode(text)  # Eliminar tildes
    if rm_s:
        if text.endswith('s'):
            text = text[:-1]  # Remover 's' al final
    return text
