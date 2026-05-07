from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import pandas as pd

load_dotenv()

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
DBNAME = os.getenv("DBNAME")


db_url = f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}/{DBNAME}'
engine = create_engine(db_url, echo=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def making_to_sql(df: pd.DataFrame, table_name: str):
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )


def upload_csv(train_path: str, test_path: str):
    making_to_sql(pd.read_csv(train_path), table_name='cars_train')
    making_to_sql(pd.read_csv(test_path),  table_name='cars_test')


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    upload_csv('train_df_path', 'test_df_path')

        



