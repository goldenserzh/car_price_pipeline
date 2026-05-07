from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import pandas as pd

load_dotenv()

DB_USERNAME = os.getenv("DB_USERNAME")
PASSWORD    = os.getenv("DB_PASSWORD")
HOST        = os.getenv("DB_HOST")
DBNAME      = os.getenv("DB_NAME")


db_url = f'postgresql+psycopg2://{DB_USERNAME}:{PASSWORD}@{HOST}/{DBNAME}'
engine = create_engine(db_url, echo=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def making_to_sql(df: pd.DataFrame, table_name: str) -> None:
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )

def connect_sql(table_train: str = 'cars_train', table_test: str = 'cars_test') -> tuple[pd.DataFrame, pd.DataFrame]:
    with engine.connect() as conn:
        df_train = pd.read_sql(f'SELECT * FROM {table_train}', conn)
        df_test  = pd.read_sql(f'SELECT * FROM {table_test}', conn)
    return df_train, df_test


def upload_csv(train_path: str, test_path: str) -> None:
    making_to_sql(pd.read_csv(train_path), table_name='cars_train')
    making_to_sql(pd.read_csv(test_path),  table_name='cars_test')




if __name__ == "__main__":
    upload_csv('train_df_path', 'test_df_path')

        



