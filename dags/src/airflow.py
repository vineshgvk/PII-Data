"""
The Airflow Dag for the preprocessing datapipeline
"""

# Import necessary libraries and modules
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from src.data_loader import load_data
from src.missing_values_handler import handle_missing
from src.duplicates_handler import remove_duplicates
from src.labelencoder.py import label_encode
from src.tokenizer.py import tokenize


# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')
conf.set('core', 'enable_parquet_xcom', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 11, 9),
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
    python_callable=load_data,
    op_kwargs={
        'excel_path': '{{ ti.xcom_pull(task_ids="  ") }}',
    },
    dag=dag,
)
# Task to handle missing values, depends on load_data_task
handle_missing_task = PythonOperator(
    task_id='missing_values_task',
    python_callable=handle_missing,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="load_data_task") }}',
    },
    dag=dag,
)

# Task to handle duplicates, depends on missing_values_task
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates_task',
    python_callable=remove_duplicates,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="handle_missing_task") }}',
    },
    dag=dag,
)

# Task to label_encode, depends on handle_duplicates
label_encode_task = PythonOperator(
    task_id='label_encode_task',
    python_callable=label_encode,
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="removing_duplicates_task") }}',
    },
    dag=dag,
)

# Task to tokenize, depends on label_encode
tokenize_task = PythonOperator(
    task_id='tokenize_task',
    python_callable=tokenize
    op_kwargs={
        'input_picle_path': '{{ ti.xcom_pull(task_ids="label_encode_task") }}',
    },
    dag=dag,
)

# Set task dependencies
load_data_task >> handle_missing_task >> remove_duplicates_task \
>> label_encode_task >> tokenize_task 

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()