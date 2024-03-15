"""
The Airflow Dag for the preprocessing datapipeline
"""

# Import necessary libraries and modules
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.models import Variable
from src.data_download import load_data_from_gcp_and_save_as_json
from src.missing_values import naHandler
from src.duplicates import dupeRemoval
from src.resampling import process_data_and_save
from src.label_encoder import load_and_transform_labels_from_json
from src.tokenize_data import tokenize_data
# from airflow.operators.email_operator import EmailOperator
# from airflow.operators.dagrun_operator import TriggerDagRunOperator
# from airflow.utils.trigger_rule import TriggerRul

# Define default arguments for your DAG
# Calculate the current time
now = datetime.now()

# Adjust the start_date to be one minute before the current time
start_date = now - timedelta(minutes=1)

default_args = {
    'owner': 'PII_Detection_T4',
    'start_date': start_date,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance named 'datapipeline'
dag = DAG(
    'PII-datapipeline',
    default_args=default_args,
    description='Airflow DAG for the datapipeline',
    schedule_interval=timedelta(days=1),  # Example: run daily
    catchup=False,
)

# Task to load data from source
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_from_gcp_and_save_as_json,
    # #op_kwargs={
    #     'json_path': '{{ var.value.json_path }}',  # Use Airflow Variables or directly specify the path
    # },
    dag=dag,
)

# Task to handle missing values
handle_missing_values_task = PythonOperator(
    task_id='missing_values_task',
    python_callable=naHandler,
    op_args=[load_data_task.output],
    # op_kwargs={
    #     'missing_inputPath': '{{ var.value.local_file_path}}',  # Update based on your variable names or direct path
    #     'missing_val_outputPath': '{{ var.value.OutPklPath }}',
    # },
    dag=dag,
)

# Task to handle duplicates
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates_task',
    python_callable=dupeRemoval,
    op_kwargs={
        'dup_inputPath': '{{ var.value.missing_val_outputPath }}',
        'dup_outputPath': '{{ var.value.outPklPath }}',
    },
    dag=dag,
)

# Task to resample the data
resample_data_task = PythonOperator(
    task_id='resample_data_task',
    python_callable=process_data_and_save,
    op_kwargs={
        'input_file_path': '{{ var.value.dup_outPklPath }}', 
        'resamp_outputPath': '{{ var.value.outPklPath }}', # Adjust based on actual output/input path
    },
    dag=dag,
)

# Task for label encoding
label_encode_task = PythonOperator(
    task_id='label_encode_task',
    python_callable=load_and_transform_labels_from_json,
    op_kwargs={
        'input_json_path': '{{ var.value.resamp_outputPath }}',
        'output_json_path': '{{ var.value.OUTPUT_LABELS_JSON_PATH  }}',
    },
    dag=dag,
)

# Task to tokenize data
tokenize_data_task = PythonOperator(
    task_id='tokenize_data_task',
    python_callable=tokenize_data,
    op_kwargs={
        'data_path': '{{ var.value.resamp_outputPath}}',
        'label2id_path': '{{ var.value.output_json_path }}',
        'output_tokenized_json_path': '{{ var.value.OUTPUT_TOKENIZED_JSON_PATH  }}',
        'model_path': 'microsoft/deberta-v3-base',  # Example model path, adjust as needed
        'max_inference_length': 512,  # Example max length, adjust as needed
    },
    dag=dag,
)

# Define task dependencies
load_data_task >> handle_missing_values_task >> remove_duplicates_task >> resample_data_task \
>> label_encode_task >> tokenize_data_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
