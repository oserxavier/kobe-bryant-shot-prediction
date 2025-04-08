# /code/data_prep/preparacao_dados.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# Inicializa o tracking do MLflow
mlflow.set_experiment("EngenhariaML")
with mlflow.start_run(run_name="PreparacaoDados"):

    # Carregando o dataset
    df = pd.read_parquet("data/raw/dataset_kobe_dev.parquet")


    # Selecionando as colunas relevantes

colunas_utilizadas = [
    "lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"
]


df = df[colunas_utilizadas]

    # Removendo valores nulos
df = df.dropna()

print(f"Dimensão final do dataset: {df.shape}")


    # Salvando dataset filtrado
df.to_parquet("data/processed/data_filtered.parquet", index=False)

    # Separando X e y
X = df.drop("shot_made_flag", axis=1)
y = df["shot_made_flag"]

    # Divisão estratificada em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Salvando arquivos
X_train.assign(shot_made_flag=y_train).to_parquet("data/processed/base_train.parquet", index=False)
X_test.assign(shot_made_flag=y_test).to_parquet("data/processed/base_test.parquet", index=False)

    # Registrando métricas no MLflow
mlflow.log_param("test_size", 0.2)
mlflow.log_metric("total_linhas", df.shape[0])
mlflow.log_metric("train_linhas", X_train.shape[0])
mlflow.log_metric("test_linhas", X_test.shape[0])
