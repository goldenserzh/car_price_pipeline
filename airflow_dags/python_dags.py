from airflow.sdk import DAG, task
from airflow.providers.standart.operators.python import PythonOperator
from datetime import datetime


with DAG(dag_id='take data', start_date=datetime.now(), schedule = " 0 0 * *") as dag:
    task_1 = PythonOperator(task_id = 'taking_data', python_command = '')
