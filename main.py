from src.model import Modeling
from src.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd



"""Задача: Создать аркестрацию apache airflow,
 логирование mlflow, обернуть как это как микросервис"""

def pulling_data():       
    data = pd.read_csv("root to the data ->")
    df = Preprocessing(data).make_preprocessing()
    return df

def 
X = df.drop('price', axis=1)
y = df['price']
cat_features = df.select_dtypes(['object']).columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Modeling()
model.find_hyperparams(X_train, X_val, y_train, y_val, cat_features, n_trials=5)
model.model_fit_predict(X_train, X_val, X_test, y_train, y_val, y_test, cat_features)
