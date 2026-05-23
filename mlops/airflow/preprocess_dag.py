from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import sys
import pandas as pd

default_args = {
    'owner': 'nil',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

S3_BUCKET = "r-mlops-bucket-12-1-1-22209764"
TMP_DIR = '/opt/airflow/dags/temp_files'

def download_events():
    os.makedirs(TMP_DIR, exist_ok=True)
    hook = S3Hook(aws_conn_id='s3_connection')
    content = hook.read_key(key='nil_project/raw_data/events.csv', bucket_name=S3_BUCKET)
    local_path = os.path.join(TMP_DIR, 'events.csv')
    with open(local_path, 'w') as f:
        f.write(content)
    print(f"Downloaded events to {local_path}")

def download_item_props_part1():
    os.makedirs(TMP_DIR, exist_ok=True)
    hook = S3Hook(aws_conn_id='s3_connection')
    content = hook.read_key(key='nil_project/raw_data/item_properties_part1.csv', bucket_name=S3_BUCKET)
    local_path = os.path.join(TMP_DIR, 'item_properties_part1.csv')
    with open(local_path, 'w') as f:
        f.write(content)
    print(f"Downloaded props part1 to {local_path}")

def download_item_props_part2():
    os.makedirs(TMP_DIR, exist_ok=True)
    hook = S3Hook(aws_conn_id='s3_connection')
    content = hook.read_key(key='nil_project/raw_data/item_properties_part2.csv', bucket_name=S3_BUCKET)
    local_path = os.path.join(TMP_DIR, 'item_properties_part2.csv')
    with open(local_path, 'w') as f:
        f.write(content)
    print(f"Downloaded props part2 to {local_path}")

def merge_item_properties():
    props1_path = os.path.join(TMP_DIR, 'item_properties_part1.csv')
    props2_path = os.path.join(TMP_DIR, 'item_properties_part2.csv')
    
    df1 = pd.read_csv(props1_path)
    df2 = pd.read_csv(props2_path)
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    merged_path = os.path.join(TMP_DIR, 'item_properties_merged.csv')
    df_merged.to_csv(merged_path, index=False)
    
    print(f"Merged properties saved to {merged_path}")

def run_preprocessing():
    events_path = os.path.join(TMP_DIR, 'events.csv')
    props_merged_path = os.path.join(TMP_DIR, 'item_properties_merged.csv')
    
    sys.path.insert(0, '/opt/airflow/dags')
    from prepare_features import prepare
    
    out_item_feats = os.path.join(TMP_DIR, 'item_features.parquet')
    out_train = os.path.join(TMP_DIR, 'data_for_training.parquet')
    
    prepare(events_path, props_merged_path, out_item_feats, out_train, window_hours=24)
    
    print("Preprocessing completed successfully")

def upload_item_features():
    local_path = os.path.join(TMP_DIR, 'item_features.parquet')
    hook = S3Hook(aws_conn_id='s3_connection')
    hook.load_file(
        filename=local_path,
        key='nil_project/processed_data/item_features.parquet',
        bucket_name=S3_BUCKET,
        replace=True
    )
    print("Uploaded item_features.parquet to S3")

def upload_train_data():
    local_path = os.path.join(TMP_DIR, 'data_for_training.parquet')
    hook = S3Hook(aws_conn_id='s3_connection')
    hook.load_file(
        filename=local_path,
        key='nil_project/processed_data/data_for_training.parquet',
        bucket_name=S3_BUCKET,
        replace=True
    )
    print("Uploaded data_for_training.parquet to S3")

with DAG(
    'ecom_preprocessing',
    default_args=default_args,
    description='Preprocess e-commerce data',
    schedule='@daily',
    catchup=False,
    tags=['ecom', 'preprocessing']
) as dag:
    
    download_events_task = PythonOperator(
        task_id='download_events',
        python_callable=download_events
    )
    
    download_props1_task = PythonOperator(
        task_id='download_item_props_part1',
        python_callable=download_item_props_part1
    )
    
    download_props2_task = PythonOperator(
        task_id='download_item_props_part2',
        python_callable=download_item_props_part2
    )
    
    merge_props_task = PythonOperator(
        task_id='merge_item_properties',
        python_callable=merge_item_properties
    )
    
    preprocess_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing
    )
    
    upload_feats_task = PythonOperator(
        task_id='upload_item_features',
        python_callable=upload_item_features
    )
    
    upload_train_task = PythonOperator(
        task_id='upload_train_data',
        python_callable=upload_train_data
    )
    
    [download_events_task, download_props1_task, download_props2_task] >> merge_props_task >> preprocess_task >> [upload_feats_task, upload_train_task]