from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from models import Base
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import pandas as pd

load_dotenv(dotenv_path = '...')

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
DBNAME = os.getenv("DBNAME")

try:
    db_url = f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}/{DBNAME}'
    engine = create_engine(db_url, echo=True)
    
except:
    raise ValueError("Нет подулючения к бд")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def making_to_sql():
    pass


def upload_csv(csv_path: str, table_name:str):
    df_train = pd.read_csv("Путь до triain_df")
    df_test = pd.read_csv('Путь до test_df')
    df_

