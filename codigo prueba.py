from funciones import load_comp_data, to_ralas, process_text, random_search
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from inject import config

## Cargar los datos de competencia
df = load_comp_data(config("PROJECT_ROOT")+'/data', sample_frac=0.2)

## Realizar ingeniería de atributos

# Cambiar product_id y user_id de float a object
df['product_id'] = df['product_id'].astype('object')
df['user_id'] = df['user_id'].astype('object')

# Sacar variables sin variabilidad
df["accepts_mercadopago"].value_counts()
cols_to_delete = ["accepts_mercadopago", "site_id", "benefit", "etl_version"]
df = df.drop(columns=cols_to_delete)

# Transformar la fecha a timestamp
df['print_server_timestamp'] = pd.to_datetime(df['print_server_timestamp'])
# Crear variables a partir de la fecha
df['print_day'] = df['print_server_timestamp'].dt.day
df['print_month'] = df['print_server_timestamp'].dt.month
df['print_day_month'] = df['print_day'].astype(str) + '-' + df['print_month'].astype(str)
df["print_week"] = df["print_server_timestamp"].dt.isocalendar().week
df["print_hour"] = df["print_server_timestamp"].dt.hour
df["print_weekday"] = df['print_server_timestamp'].dt.weekday < 5  # 0 para el lunes. 6 para el domingo
df['print_day_of_week'] = df['print_server_timestamp'].dt.day_name()
df['print_server_numeric'] = df['print_server_timestamp'].values.astype("float64")

# Transformar los missings NO aleatorios a -1
df["is_pdp"] = df["is_pdp"].astype('float').fillna(-1).astype('int')
df['user_id'] = np.where(df['user_id'].isna(), "-1", df['user_id'])

# Crear algunos ratios
df["conversion_item_30d"] = df["total_orders_item_30days"] / df["total_visits_item"]
df['conversion_domain_30d'] = df['total_orders_domain_30days'] / df['total_visits_domain']

# Unir dos variables relevantes
df["offset_print_potition"] = df["offset"] * df["print_position"]

# Estandarizar la garantía
pattern = r'(?:\w+\s+)?(\d+(?:\.\d+)?)\s+(\w+)'
df[["warranty_qty", "warranty_time_type"]] = df['warranty'].str.extract(pattern)
df["warranty_time_type"].value_counts()  # Tenemos problemas de formato
df['warranty_time_type'] = np.where(df['warranty_time_type'].isna(), "No", df['warranty_time_type'])  # Primero transformar los missings
df['warranty_time_type'] = df['warranty_time_type'].apply(process_text)
df["warranty_time_type"].value_counts()

df["warranty_qty"] = df["warranty_qty"].astype(np.float64)
df["warranty_qty"] = np.where(df["warranty_qty"].isna(), -1, df["warranty_qty"])  # Transformar los missings

# Aprovechar el texto
df["title"] = df["title"].apply(lambda x: process_text(x, rm_s=False))  # Limpiar el texto

df['title_oficial'] = df['title'].str.contains("oficial") * 1  # Buscar si la palabra "oficial" está dentro del título
df['title_liquidacion'] = df['title'].str.contains("liquidacion") * 1
df['title_envio'] = df['title'].str.contains("envio") * 1
df['title_nuevo'] = df['title'].str.contains("nuevo") * 1
df['title_nueva'] = df['title'].str.contains("nueva") * 1

df['title_oficial'].value_counts(normalize=True)
df['title_liquidacion'].value_counts(normalize=True)
df['title_envio'].value_counts(normalize=True)
df['title_nuevo'].value_counts(normalize=True)
df['title_nueva'].value_counts(normalize=True)

# Otra forma de aprovechar el texto
df["title_len"] = df["title"].str.len()

# Transformar booleanos en int
df["free_shipping"] = np.where(df["free_shipping"], 1, 0)
df["print_weekday"] = np.where(df["print_weekday"], 1, 0)

# Hacer OHE de la columna tags
# Obtener todos los valores únicos de la columna tags. Ojo, parece una columna de listas, pero en verdad es string.
unique_values = set()
for row in df['tags']:
    unique_values.update(row[1:-1].split(","))  # Con [1:-1] sacamos los corchetes del string del principio y final
# Crear columnas dummy para cada valor único
for value in unique_values:
    df[value] = df['tags'].apply(lambda x: 1 if value in x else 0)

## Seleccionar las columnas relevantes
to_keep_numeric = df.select_dtypes(include="number").columns.tolist()
to_do_ohe = ["platform", "fulfillment", "logistic_type", "listing_type_id", "category_id", "domain_id",
             "product_id", "user_id", "print_day_of_week"]
to_keep_boolean = ["conversion"]

# Realizar One-Hot Encoding (OHE) en las columnas categóricas
df_categorical = pd.get_dummies(df[to_do_ohe], sparse=True, dummy_na=True)

# Columnas a utilizar para el modelo. Permitanme incluir domain_id y print_day_month.
df = pd.concat([df[["train_eval", "domain_id", "print_day_month"] + to_keep_numeric + to_keep_boolean], df_categorical],
               axis=1)

# Convertir las columnas booleanas a 0 y 1, manteniendo los valores NaN
for col in to_keep_boolean:
    df[col] = np.where(df[col].isna(), np.nan, df[col].astype(float))
    to_keep_numeric.append(col)

# Separar el conjunto de entrenamiento y evaluación
df_train = df.loc[df["train_eval"] == "train"]
df_train = df_train.drop(columns=["ROW_ID", "train_eval"])

df_eval = df.loc[df["train_eval"] == "eval"]
df_eval = df_eval.drop(columns=["conversion", "train_eval"])

# Separar un conjunto de validación del conjunto de entrenamiento (Esto se podría hacer mejor)
df_valid = df_train.iloc[:10000, :].copy()
df_train = df_train.iloc[10000:, :].copy()

# Eliminar domain_id y print_day_month para el modelo
df_train = df_train.drop(columns=["domain_id", "print_day_month"])
df_valid = df_valid.drop(columns=["domain_id", "print_day_month"])
df_eval = df_eval.drop(columns=["domain_id", "print_day_month"])

dtrain_rala, df_train_names = to_ralas(df_train)
dvalid_rala, df_valid_names = to_ralas(df_valid)
deval_rala, df_eval_names = to_ralas(df_eval)

# Crear objetos DMatrix para XGBoost
dtrain = xgb.DMatrix(dtrain_rala[:, df_train_names != "conversion"],
                     label=dtrain_rala[:, df_train_names == "conversion"].todense().squeeze(),
                     feature_names=df_train_names[df_train_names != "conversion"].tolist())

dvalid = xgb.DMatrix(dvalid_rala[:, df_valid_names != "conversion"],
                     label=dvalid_rala[:, df_valid_names == "conversion"].todense().squeeze(),
                     feature_names=df_valid_names[df_valid_names != "conversion"].tolist())

deval = xgb.DMatrix(deval_rala[:, df_eval_names != "ROW_ID"],
                    feature_names=df_eval_names[df_eval_names != "ROW_ID"].tolist())

# Definir los parámetros del modelo XGBoost
watchlist = [(dtrain, "train"), (dvalid, "validation")]

# Definir el espacio de hiperparámetros para la búsqueda aleatoria
param_dist = {
    "objective": ["binary:logistic"],
    "eval_metric": ["auc"],
    "max_depth": [50],
    "eta": [0.01],
    "gamma": [10],
    "colsample_bytree": [0.8],
    "min_child_weight": [10],
    "subsample": [0.8],
    "num_boost_round": [50]
}

# Probar solo una combinación
best_params, best_score, best_model, exp_results = random_search(param_dist, dtrain, watchlist, n_iter=1)

# Calcular la importancia de las características
importance = best_model.get_score(importance_type='gain')

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, importance_type='gain', max_num_features=18)
plt.title('Feature Importance')
plt.show()

y_preds_eval = best_model.predict(deval)

# Crear el archivo de envío para Kaggle
submission_df = pd.DataFrame({"ROW_ID": deval_rala[:, df_eval_names == "ROW_ID"].toarray().squeeze(),
                              "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)



try:
    submission_df.to_csv(config('PROJECT_ROOT') + "/submission.csv", index=False)
    print("Archivo CSV guardado correctamente.")
except Exception as e:
    print(f"Error al guardar el archivo CSV: {str(e)}")

    print(submission_df.head())