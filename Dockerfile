FROM apache/airflow:2.8.3
# Install additional required packages
RUN pip install --no-cache-dir transformers torch sentencepiece tensorboard scikit-learn numpy pandas google-cloud-storage datasets streamlit matplotlib seaborn mlflow DateTime accelerate