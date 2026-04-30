from catboost import CatboostRegressor
from optuna.pruners import HyperbandPruner
import optuna
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error,
                              root_mean_squared_error)

from sklearn.model_selection import train_test_split
import pandas as pd



class Modeling:

  HYPERPARAMS = {}

  def __init__(self, model:CatBoostRegressor) -> None:
     self.model = model
     self.test_pred = None

  @staticmethod
  def show_metrics(y_pred, y_true):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"rmse: {rmse}")

  def find_hyperparams(self, df: pd.DataFrame, n_trials: int = 10 ) -> None:
    X = df.drop('price', axis=1)
    y = df['price']

    cat_features= df.select_dtypes(['object']).columns.to_list()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def objective(trial: optuna.Trial) -> float:
      params = {
                'iterations':        trial.suggest_int('iterations', 500, 2000),
                'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'depth':             trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength':   trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
                'border_count':      trial.suggest_int('border_count', 32, 255),
                'verbose': 100,
                'random_seed': 42,
                'loss_function': 'RMSE',
            }

      model = CatBoostRegressor(**params)
      model.fit(X_train, y_train, cat_features=cat_features, eval_set = (X_val, y_val), verbose=100)
      y_pred = model.predict(X_val)
      return mean_absolute_error(y_val, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    self.HYPERPARAMS = study.best_params
    print(f"Best params: {self.HYPERPARAMS}")
    print(f"Best score: {study.best_value}")


  def model_fit_predict(self, df: pd.DataFrame, ) -> None:
    self.model = self.model(**self.HYPERPARAMS)
    X = df.drop('price', axis=1)
    cat_features = df.select_dtypes(['object']).columns.to_list()
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if len(self.HYPERPARAMS) == 0:
      print("Подбор гиперпараметров ещё не реализован")
      self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=100)
      self.test_pred = self.model.predict(X_val)
      self.show_metrics(self.test_pred, y_val)
    else:
      self.model = self.model(**self.HYPERPARAMS)
      self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=100)
      self.test_pred = self.model.predict(X_val)
      self.show_metrics(self.test_pred, y_val)




