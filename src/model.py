from catboost import CatBoostRegressor
from optuna.pruners import HyperbandPruner
import optuna
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             root_mean_squared_error)

import mlflow


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

    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"rmse: {rmse}")

  def find_hyperparams(self, X_train, X_val, y_train, y_val, cat_features, n_trials: int = 10) -> None:

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
    print(f"Best params: {self.HYPERPARAMS}")
    print(f"Best score: {study.best_value}")


  def model_fit_predict(self, X_train, X_val, X_test, y_train, y_val, y_test, cat_features) -> None:

    if len(self.HYPERPARAMS) == 0:
      print("Подбор гиперпараметров ещё не выполнен")
      with mlflow.start_run():
        self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=100)
        self.test_pred = self.model.predict(X_test)
        self.show_metrics(self.test_pred, y_test)
    else:
      self.model = CatBoostRegressor(**self.HYPERPARAMS)
      with mlflow.start_run():
        self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=100)
        self.test_pred = self.model.predict(X_test)
        self.show_metrics(self.test_pred, y_test)
