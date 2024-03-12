"""
The Airflow Dag for the preprocessing datapipeline
"""

# Import necessary libraries and modules
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from src.data_download.py import load_data_from_gcp_and_save_as_pkl
from src.missing_values.py import naHandler
from src.duplicates.py import dupeRemoval
from src.labelencoder.py import load_and_transform_labels_from_pkl
from src.tokenizer.py import tokenize_data


# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')
conf.set('core', 'enable_parquet_xcom', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 3, 5),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Create a DAG instance named 'datapipeline' with the defined default arguments
dag = DAG(
    'datapipeline',
    default_args=default_args,
    description='Airflow DAG for the datapipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)
# Task to load data from source
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_from_gcp_and_save_as_pkl,
    op_kwargs={
        'excel_path': '{{ ti.xcom_pull(task_ids="  ") }}',
    },
    dag=dag,
)
# Task to handle missing values, depends on load_data_task
handle_missing_values_task = PythonOperator(
    task_id='missing_values_task',
    python_callable=naHandler,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="load_data_task") }}',
    },
    dag=dag,
)

# Task to handle duplicates, depends on missing_values_task
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates_task',
    python_callable=dupeRemoval,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="handle_missing_values_task") }}',
    },
    dag=dag,
)

# Task to label_encode, depends on handle_duplicates
label_encoder_task = PythonOperator(
    task_id='label_encode_task',
    python_callable=load_and_transform_labels_from_pkl,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="removing_duplicates_task") }}',
    },
    dag=dag,
)

# Task to tokenize, depends on label_encoder
tokenize_task = PythonOperator(
    task_id='tokenize_task',
    python_callable=tokenize_data,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="label_encoder_task") }}',
    },
    dag=dag,
)

# Set task dependencies
load_data_task >> handle_missing_values_task >> remove_duplicates_task \
>> label_encoder_task >> tokenize_task 

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()