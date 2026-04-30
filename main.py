from src.model import Modeling
from src.preprocessing import Preprocessing
import pandas as pd
# from fastapi import

"""Задача: Создать аркестрацию apache airflow,
 логирование mlflow, обернуть как это как микросервис"""

data = pd.read_csv("root to the data ->")
df = Preprocessing(data)

model = Modeling()
model.find_hyperparams(df, n_trials=5)
model.model_fit_predict(df)


