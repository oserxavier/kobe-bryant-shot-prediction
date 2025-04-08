# /code/training/treinamento_modelos.py

import pandas as pd
import mlflow
from pycaret.classification import *

# Inicia o tracking
mlflow.set_experiment("EngenhariaML")

# Carrega os dados processados
df_train = pd.read_parquet("data/processed/base_train.parquet")
df_test = pd.read_parquet("data/processed/base_test.parquet")

# Inicia uma run no MLflow para a regressão logística
with mlflow.start_run(run_name="Treinamento_Regression"):
    # Setup PyCaret
    s = setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        log_experiment=True,
        experiment_name="EngenhariaML",
        silent=True,
        use_gpu=False
    )
    
    model = create_model("lr")  # regressão logística
    tuned = tune_model(model)
    final_model = finalize_model(tuned)

    # Avaliação na base de teste
    predictions = predict_model(final_model, data=df_test)
    metrics = pull()
    log_loss_val = metrics.loc[metrics['Metric'] == 'Log Loss', 'Value'].values[0]
    
    mlflow.log_metric("log_loss", log_loss_val)

# Agora para a Árvore de Decisão
with mlflow.start_run(run_name="Treinamento_DecisionTree"):
    s = setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        log_experiment=True,
        experiment_name="EngenhariaML",
        silent=True,
        use_gpu=False
    )

    model = create_model("dt")  # árvore de decisão
    tuned = tune_model(model)
    final_model = finalize_model(tuned)

    predictions = predict_model(final_model, data=df_test)
    metrics = pull()
    
    log_loss_val = metrics.loc[metrics['Metric'] == 'Log Loss', 'Value'].values[0]
    f1_val = metrics.loc[metrics['Metric'] == 'F1', 'Value'].values[0]

    mlflow.log_metric("log_loss", log_loss_val)
    mlflow.log_metric("f1_score", f1_val)
