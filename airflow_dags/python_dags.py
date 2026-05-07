from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime
import sys
import pandas as pd
sys.path.insert(0, 'абсолютный путь дирректории')
from src.sql_connection import connect_sql, making_to_sql
from src.preprocessing import Preprocessing
from src.model import Modeling


RAW_TRAIN_TMP       = '/tmp/cars_train_raw.parquet'
RAW_TEST_TMP        = '/tmp/cars_test_raw.parquet'
PROCESSED_TRAIN_TMP = '/tmp/cars_train_processed.parquet'
PROCESSED_TEST_TMP  = '/tmp/cars_test_processed.parquet'


def fetch_data(table_train: str, table_test: str, train_path: str, test_path: str):
    df_train, df_test = connect_sql(table_train, table_test)
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)


def run_preprocessing():
    df_train = pd.read_parquet(RAW_TRAIN_TMP)
    df_test  = pd.read_parquet(RAW_TEST_TMP)

    df_train = Preprocessing(df_train).make_preprocessing()
    df_test  = Preprocessing(df_test).make_preprocessing()

    df_train.to_parquet(PROCESSED_TRAIN_TMP, index=False)
    df_test.to_parquet(PROCESSED_TEST_TMP, index=False)

    making_to_sql(df_train, table_name='cars_train_processed')
    making_to_sql(df_test,  table_name='cars_test_processed')


def modelling_cb():
    df_train, df_test = connect_sql('cars_train_processed', 'cars_test_processed')
    cat_features = df_train.select_dtypes(['object']).columns.to_list()
    model = Modeling()
    model.find_hyperparams(df_train, cat_features)
    model.model_fit_predict(df_train, df_test, cat_features)




with DAG(dag_id="upload_csv_to_sql", start_date=datetime(2026, 1, 1)) as bash_dag:
    run_sql_con = BashOperator(
        task_id='upload_csv',
        bash_command='cd /абсолютный/путь/дирректории && python -m src.sql_connection'
    )


with DAG(dag_id='sql_to_processed', start_date=datetime(2026, 1, 1), schedule='0 0 * * *') as from_sql_dag:
    taking_data = PythonOperator(
        task_id='taking_data',
        python_callable=fetch_data,
        op_kwargs={
            'table_train': 'cars_train',
            'table_test':  'cars_test',
            'train_path':  RAW_TRAIN_TMP,
            'test_path':   RAW_TEST_TMP,
        }
    )
    preprocessing = PythonOperator(task_id='make_preprocessing', python_callable=run_preprocessing)

    taking_data >> preprocessing


with DAG(dag_id='learning_model', start_date=datetime(2026, 1, 1), schedule='0 0 * * *') as modeling:
    taking_ready_data = PythonOperator(
        task_id='taking_ready_data',
        python_callable=fetch_data,
        op_kwargs={
            'table_train': 'cars_train_processed',
            'table_test':  'cars_test_processed',
            'train_path':  PROCESSED_TRAIN_TMP,
            'test_path':   PROCESSED_TEST_TMP,
        }
    )

    learning_model = PythonOperator(
        task_id = 'leaning_model',
        python_callable = modelling_cb,
    )

    taking_ready_data >> learning_model






