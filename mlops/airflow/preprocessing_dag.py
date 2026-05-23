from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

default_args = {
    'owner': 'nil',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def upload_to_s3(local_path, key):
    hook = S3Hook(aws_conn_id='s3_connection')
    hook.load_file(filename=local_path, key=key, bucket_name='storage.yandexcloud.net', replace=True)

def download_from_s3(key, local_dir, **context):
    hook = S3Hook(aws_conn_id='s3_connection')
    os.makedirs(local_dir, exist_ok=True)
    # download_file сам создаст файл с именем из key
    hook.download_file(key=key, bucket_name='r-mlops-bucket-12-1-1-22209764', local_path=local_dir)
    print(f"Downloaded {key} to {local_dir}")

def run_preprocessing(**context):
    tmpdir = '/opt/airflow/dags/temp_files'
    os.makedirs(tmpdir, exist_ok=True)
    
    # Передаём директорию, а не полный путь к файлу
    download_from_s3('nil_project/raw_data/events.csv', tmpdir, **context)
    download_from_s3('nil_project/raw_data/item_properties_part1.csv', tmpdir, **context)
    
    events_path = os.path.join(tmpdir, 'events.csv')
    item_props_path = os.path.join(tmpdir, 'item_properties_part1.csv')
    
    sys.path.insert(0, '/opt/airflow/dags')
    from prepare_features import prepare
    
    out_item_feats = os.path.join(tmpdir, 'item_features.parquet')
    out_train = os.path.join(tmpdir, 'data_for_training.parquet')
    
    prepare(events_path, item_props_path, out_item_feats, out_train, window_hours=24)
    
    upload_to_s3(out_item_feats, 'nil_project/processed_data/item_features.parquet', **context)
    upload_to_s3(out_train, 'nil_project/processed_data/data_for_training.parquet', **context)
    
    print("Preprocessing completed successfully")

with DAG(
    'ecom_preprocessing',
    default_args=default_args,
    description='Preprocess e-commerce data',
    schedule='@daily',
    catchup=False,
) as dag:
    
    preprocess_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing,
    )