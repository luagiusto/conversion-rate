import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from funciones import load_comp_data, to_ralas, process_text, random_search
from decouple import config
from sklearn.model_selection import train_test_split


# Cargar los datos de competencia
df = load_comp_data(config("PROJECT_ROOT") + '/data', sample_frac=0.2)


# Convertir columnas a tipo 'object'
df['product_id'] = df['product_id'].astype('object')
df['user_id'] = df['user_id'].astype('object')

# Eliminar columnas innecesarias
cols_to_delete = ["accepts_mercadopago", "site_id", "benefit", "etl_version"]
df = df.drop(columns=cols_to_delete)

# Convertir la columna de fecha a datetime
df['print_server_timestamp'] = pd.to_datetime(df['print_server_timestamp'])
df['print_day'] = df['print_server_timestamp'].dt.day
df['print_month'] = df['print_server_timestamp'].dt.month
df['print_day_month'] = df['print_day'].astype(str) + '-' + df['print_month'].astype(str)
df["print_hour"] = df["print_server_timestamp"].dt.hour
df["print_weekday"] = df['print_server_timestamp'].dt.weekday < 5
df['print_day_of_week'] = df['print_server_timestamp'].dt.day_name()
df['print_server_numeric'] = df['print_server_timestamp'].values.astype("float64")


# df printmonth unique values
df['conversion'].unique()

# Plot conversion by print_server_timestamp
df['conversion'] = df['conversion'].astype(float)
# print_day_month to datetime
df['print_day_month'] = pd.to_datetime(df['print_day_month'], format='%d-%m')
mean_conversion_by_day_month = df.groupby('print_day_month')['conversion'].mean().reset_index()


df["is_pdp"] = df["is_pdp"].astype('float').fillna(-1).astype('int')
df['user_id'] = np.where(df['user_id'].isna(), "-1", df['user_id'])

df["conversion_item_30d"] = df["total_orders_item_30days"] / df["total_visits_item"]
df['conversion_domain_30d'] = df['total_orders_domain_30days'] / df['total_visits_domain']
df["offset_print_potition"] = df["offset"] * df["print_position"]

# Convertir la columna 'warranty' a string
df['warranty'] = df['warranty'].astype(str)

# Extraer patrones de la columna 'warranty'
pattern = r'(?:\w+\s+)?(\d+(?:\.\d+)?)\s+(\w+)'
df[["warranty_qty", "warranty_time_type"]] = df['warranty'].str.extract(pattern)
df['warranty_time_type'] = np.where(df['warranty_time_type'].isna(), "No", df['warranty_time_type'])
df['warranty_time_type'] = df['warranty_time_type'].apply(process_text)
df["warranty_qty"] = df["warranty_qty"].astype(np.float64)
df["warranty_qty"] = np.where(df["warranty_qty"].isna(), -1, df["warranty_qty"])

# Procesar el texto de la columna 'title'
df["title"] = df["title"].apply(lambda x: process_text(x, rm_s=False))
df['title_oficial'] = df['title'].str.contains("oficial") * 1
df['title_liquidacion'] = df['title'].str.contains("liquidacion") * 1
df['title_envio'] = df['title'].str.contains("envio") * 1
df['title_nuevo'] = df['title'].str.contains("nuevo") * 1
df['title_nueva'] = df['title'].str.contains("nueva") * 1
df["title_len"] = df["title"].str.len()

# Convertir columnas booleanas a 0 y 1
df["free_shipping"] = np.where(df["free_shipping"], 1, 0)
df["print_weekday"] = np.where(df["print_weekday"], 1, 0)

# Interacciones de atributos
df['log_price'] = np.log1p(df['price'])

# Bin-counting de categorías
category_count = df['category_id'].value_counts()
df['category_count'] = df['category_id'].map(category_count)

# Agregaciones por 'domain_id'
df['avg_price_by_domain'] = df.groupby('domain_id')['price'].transform('mean')

# Procesar etiquetas en 'tags'
unique_values = set()
for row in df['tags']:
    unique_values.update(row[1:-1].split(","))
for value in unique_values:
    df[value] = df['tags'].apply(lambda x: 1 if value in x else 0)

# drop columnas duplicadas
for col in df.columns:
    if df[col].nunique() == 1:
        df.drop(col, axis=1, inplace=True)
#Feature engineering
# Crear variable more_than_one_conversion_for_same_session_id
df['more_than_one_conversion_for_same_session_id'] = df['uid'].duplicated(keep=False).astype(int)

# Crear variable more_than_one_conversion_for_same_user_id
df['more_than_one_conversion_for_same_user_id'] = df['user_id'].duplicated(keep=False).astype(int)

# Mismo usuario compro 7,14,30 dias
df['same_user_bought_in_last_7_days'] = df['user_id'].isin(
    df.loc[df['print_server_timestamp'] >= df['print_server_timestamp'] - pd.Timedelta(days=7), 'user_id']).astype(
    int)

df['same_user_bought_in_last_14_days'] = df['user_id'].isin(
    df.loc[df['print_server_timestamp'] >= df['print_server_timestamp'] - pd.Timedelta(days=14), 'user_id']).astype(
    int)

df['same_user_bought_in_last_30_days'] = df['user_id'].isin(
    df.loc[df['print_server_timestamp'] >= df['print_server_timestamp'] - pd.Timedelta(days=30), 'user_id']).astype(
    int)
# Si print_hour es dentro 8 y 18
df['print_hour_between_8_and_18'] = df['print_hour'].between(8, 18).astype(int)
# log_price > 8.5 y menos que 11
df['log_price_between_8.5_and_11'] = df['log_price'].between(8.5, 11).astype(int)


#Plot distribution of variables by conversion
# Plotting the distribution of variables by conversion
# Variables to plot
#vars_to_plot = ['price', 'available_quantity', 'sold_quantity', 'log_price', 'price_per_quantity', 'category_count',
#                'avg_price_by_domain']

# Plotting the distribution of variables by conversion
#for var in df.columns:
#    try:
#        plt.figure(figsize=(12, 6))
#        df.groupby('conversion')[var].plot(kind='kde', legend=True)
#        plt.title(f'Distribution of {var} by Conversion')
#        plt.xlabel(var)
#        plt.ylabel('Density')
#        plt.legend(['No Conversion', 'Conversion'])
#        plt.grid(True)
#        plt.savefig(f'output/distribution_of_{var}_by_conversion.png')
#    except Exception as e:
#        print(f'Error plotting {var}, {str(e)}')


# Drop columna si tiene valores inf
df = df.replace([np.inf, -np.inf], np.nan)


# Separar columnas según su tipo
to_keep_numeric = df.select_dtypes(include="number").columns.tolist()
to_do_ohe = ["platform", "fulfillment", "logistic_type", "listing_type_id", "category_id", "domain_id",
             "product_id", "user_id", "warranty_time_type", "print_day_of_week"]
to_keep_boolean = ["conversion"]

# # One-hot encoding para variables categóricas
df_categorical = pd.get_dummies(df[to_do_ohe], sparse=True, dummy_na=True)

# Concatenar todas las columnas
df = pd.concat([df[["train_eval", "domain_id", "print_day_month"] + to_keep_numeric + to_keep_boolean], df_categorical],
               axis=1)

# Convertir columnas booleanas a float
for col in to_keep_boolean:
    df[col] = np.where(df[col].isna(), np.nan, df[col].astype(float))
    to_keep_numeric.append(col)

# Identificar columnas donde el valor unico es 1 y droppearlas
cols_to_drop = df.columns[df.nunique() == 1]

# Drop these columns from the dataframe
df.drop(cols_to_drop, axis=1, inplace=True)


# Drop duplicate columns
df = df.loc[:, ~df.columns.duplicated()]



# Separar conjunto de datos en entrenamiento y evaluación
df_train = df.loc[df["train_eval"] == "train"]
df_train = df_train.drop(columns=["ROW_ID", "train_eval"])

df_eval = df.loc[df["train_eval"] == "eval"]
df_eval = df_eval.drop(columns=["conversion", "train_eval"])

# split by conversion train_test_split stratified by conversion. 0.2
df_train, df_valid = train_test_split(
    df_train,
    train_size=0.8,
    stratify=df_train["conversion"]
)

df_train = df_train.drop(columns=["domain_id", "print_day_month"])
df_valid = df_valid.drop(columns=["domain_id", "print_day_month"])
df_eval = df_eval.drop(columns=["domain_id", "print_day_month"])

# Convertir los DataFrames a matrices RALA
dtrain_rala, df_train_names = to_ralas(df_train)
dvalid_rala, df_valid_names = to_ralas(df_valid)
deval_rala, df_eval_names = to_ralas(df_eval)

# Crear DMatrix para XGBoost
dtrain = xgb.DMatrix(dtrain_rala[:, df_train_names != "conversion"],
                     label=dtrain_rala[:, df_train_names == "conversion"].todense().squeeze(),
                     feature_names=df_train_names[df_train_names != "conversion"].tolist(),
                     missing='inf')

dvalid = xgb.DMatrix(dvalid_rala[:, df_valid_names != "conversion"],
                     label=dvalid_rala[:, df_valid_names == "conversion"].todense().squeeze(),
                     feature_names=df_valid_names[df_valid_names != "conversion"].tolist(),
                     missing='inf')

deval = xgb.DMatrix(deval_rala[:, df_eval_names != "ROW_ID"],
                    feature_names=df_eval_names[df_eval_names != "ROW_ID"].tolist(),
                    missing='inf')

# Configuración de parámetros y entrenamiento del modelo
watchlist = [(dtrain, "train"), (dvalid, "validation")]

param_dist = {
    "objective": ["binary:logistic"],
    "eval_metric": ["auc"],
    "max_depth": [3, 6, 9, 0],
    "eta": [0.01, 0.1, 0.2],
    "gamma": [1, 10, 100],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "num_boost_round": [100]
}

# Búsqueda de hiperparámetros
best_params, best_score, best_model, exp_results = random_search(param_dist, dtrain, watchlist, n_iter=1)

# Importancia de características
importance = best_model.get_score(importance_type='gain')

plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, importance_type='gain', max_num_features=18)
plt.title('Feature Importance')
plt.show()

# Predicciones en el conjunto de evaluación
y_preds_eval = best_model.predict(deval)

# Crear el archivo de envío para Kaggle
submission_df = pd.DataFrame({"ROW_ID": deval_rala[:, df_eval_names == "ROW_ID"].toarray().squeeze(),
                              "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)

# Guardar el DataFrame en un archivo CSV
try:
    submission_df.to_csv(config('PROJECT_ROOT') + "/output/submission.csv", index=False)
    print("Archivo CSV guardado correctamente.")
except Exception as e:
    print(f"Error al guardar el archivo CSV: {str(e)}")

print(submission_df.head())
