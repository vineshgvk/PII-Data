from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago

# Make sure to import your function. Adjust the import path as necessary.
from src.data_download import load_data_from_gcp_and_save_as_json
from src.missing_values import naHandler
from src.duplicates import dupeRemoval
from src.resampling import resample_data
from src.label_encoder import target_label_encoder
from src.tokenize_data import tokenize_data

now = datetime.now()

# Adjust the start_date to be one minute before the current time
start_date = now - timedelta(minutes=1)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    # 'start_date': start_date,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'PII_Data_Detection',
    default_args=default_args,
    description='A DAG to load data from GCP and process it',
    # schedule_interval=timedelta(days=1),
    schedule_interval="*/2 * * * *",
    catchup=False
)
load_data_task = PythonOperator(
    task_id='load_data_from_gcp',
    python_callable=load_data_from_gcp_and_save_as_json,
    dag=dag,
)


handle_missing_values_task = PythonOperator(
    task_id='missing_values_removal',
    python_callable=naHandler,
    # op_kwargs={'outputPath': '/home/vineshgvk/PII-Data/dags/processed/after_missing_values.pkl'},  # Specify the desired output path
    provide_context=True,
    dag=dag,
)

# Task to handle duplicates
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates',
    python_callable=dupeRemoval,
    # op_kwargs={
    #     'dup_inputPath': '{{ var.value.missing_val_outputPath }}',
    #     'dup_outputPath': '{{ var.value.outPklPath }}',
    # },
    provide_context=True,
    dag=dag,
)


# Task to resample the data
resample_data_task = PythonOperator(
    task_id='resample_data',
    python_callable=resample_data,
    # op_kwargs={
    #     'input_file_path': '{{ var.value.dup_outPklPath }}', 
    #     'resamp_outputPath': '{{ var.value.outPklPath }}', # Adjust based on actual output/input path
    # },
    provide_context=True,
    dag=dag,
)


# Task for label encoding
label_encode_task = PythonOperator(
    task_id='label_encoder',
    python_callable=target_label_encoder,
    # op_kwargs={
    #     'input_json_path': '{{ var.value.resamp_outputPath }}',
    #     'output_json_path': '{{ var.value.OUTPUT_LABELS_JSON_PATH  }}',
    # },
    provide_context=True,
    dag=dag,
)

# Task to tokenize data
tokenize_data_task = PythonOperator(
    task_id='tokenize_data_task',
    python_callable=tokenize_data,
    # op_kwargs={
    #     'data_path': '{{ var.value.resamp_outputPath}}',
    #     'label2id_path': '{{ var.value.output_json_path }}',
    #     'output_tokenized_json_path': '{{ var.value.OUTPUT_TOKENIZED_JSON_PATH  }}',
    #     'model_path': 'microsoft/deberta-v3-base',  # Example model path, adjust as needed
    #     'max_inference_length': 512,  # Example max length, adjust as needed
    # },
    dag=dag,
)




load_data_task >> handle_missing_values_task >>remove_duplicates_task >> resample_data_task >> label_encode_task >> tokenize_data_task

if __name__ == "__main__":
    dag.cli()
