from catboost import CatBoostRegressor
from optuna.pruners import HyperbandPruner
import optuna
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             root_mean_squared_error)

import mlflow
import logging
from sklearn.model_selection import train_test_split


class Modeling:

  def __init__(self) -> None:
     self.model = CatBoostRegressor()
     self.test_pred = None
     self.HYPERPARAMS = {}

  @staticmethod
  def show_metrics(y_pred, y_true):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    logging.info(f"mae: {mae}")
    logging.info(f"mse: {mse}")
    logging.info(f"rmse: {rmse}")

  def find_hyperparams(self, df_train, cat_features, n_trials: int = 10) -> None:
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
    def objective(trial: optuna.Trial) -> float:
      params = {
                'iterations':          trial.suggest_int('iterations', 500, 2000),
                'learning_rate':       trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'depth':               trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength':     trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
                'border_count':        trial.suggest_int('border_count', 32, 255),
                'verbose':             0,
                'random_seed':         42,
                'loss_function':       'RMSE',
                'cat_features':        cat_features,
            }

      model = CatBoostRegressor(**params)
      model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
      y_pred = model.predict(X_val)
      return mean_absolute_error(y_val, y_pred)

    study = optuna.create_study(direction='minimize', pruner=HyperbandPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    self.HYPERPARAMS = study.best_params
    logging.info(f"Best params: {self.HYPERPARAMS}")
    logging.info(f"Best score: {study.best_value}")


  def model_fit_predict(self, df_train, df_test, cat_features) -> None:
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    X_test = df_test.drop('price', axis=1)
    y_test = df_test['price']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    if not self.HYPERPARAMS:
      logging.info("Подбор гиперпараметров ещё не выполнен")
    else:
      self.model = CatBoostRegressor(**self.HYPERPARAMS)

    with mlflow.start_run():
      mlflow.log_params(self.HYPERPARAMS)
      self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=100)
      self.test_pred = self.model.predict(X_test)
      mae  = mean_absolute_error(y_test, self.test_pred)
      mse  = mean_squared_error(y_test, self.test_pred)
      rmse = root_mean_squared_error(y_test, self.test_pred)
      mlflow.log_metrics({'mae': mae, 'mse': mse, 'rmse': rmse})
      self.show_metrics(self.test_pred, y_test)
