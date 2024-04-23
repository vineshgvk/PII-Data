# Personally Identifiable Information Detection in Student Writing
[Sai Akhil Rayapudi](https://github.com/rayapudisaiakhil)
[Jayatha Chandra](https://github.com/jayathachan)
[Lahari Boni](https://github.com/LahariBoni)
[Vinesh Kumar Gande](https://github.com/vineshgvk)
[Abhigna Reddy](https://github.com/Abhignareddy7211)
[Siddharthan Singaravel](https://github.com/SiddharthanSingaravel)

# Introduction
The advent of AI heightens the risk of data and identity theft, with many scams stemming from data breaches and stolen Personally Identifiable Information (PII).  in the U.S. alone, there have been 2,691 data breaches in K–12 school districts and colleges/universities, affecting nearly 32 million records. Remarkably, 83% of these records were from post-secondary institutions, primarily due to hacking and ransomware attacks.

Post-pandemic, there has been a significant shift among educational institutions towards online teaching and the widespread adoption of digital tools for various academic activities. Between 2016 and 2020, thousands of students had their personal information compromised, including grades and Social Security numbers, leading to various forms of harm​ ​. E-Learning platforms, which have gained immense popularity due to their affordability and high-quality education, must prioritize the security of Personally Identifiable Information (PII).

These platforms are trusted by many and must prioritize the security of Personally Identifiable Information (PII), particularly as students frequently share sensitive data during their academic interactions. The protection of this data is crucial to prevent students from falling victim to scams, particularly affecting vulnerable groups like job seekers and immigrants. Ensuring the security of PII in education is not only about safeguarding individuals but also about averting legal and financial consequences for these institutions.

In 2022 alone, the education sector saw a 44% increase in cyberattacks, marking it as the most targeted industry for such incidents. This dramatic rise illustrates the urgent need for enhanced security measures.

In this project, we work on identifying Personally Identifiable Information (PII) in textual data, especially when it’s embedded in the extensive personal details often found in Student Essays. Despite the critical importance of protecting PII, detecting it within vast amounts of textual data poses a significant challenge. Manual inspection is inefficient, impractical, and laborious. Keyword searches often overlook PII's complexity and variability.

With the exponential growth of text data and the complexities of PII detection, there is an urgent need for sophisticated and automated methods. Hence, Machine learning emerges as a pivotal solution, offering the accuracy and automation required to adeptly detect and secure PII within vast textual databases.

In this project, we will explore key exploratory data analysis techniques and adopt a hybrid approach for PII detection. We plan to integrate Regex for basic detection, augment NER with deep learning for refined accuracy, and leverage transformer models for sophisticated PII analysis.

Our goals comprise data pipeline preparation, ML Model Training, and ML Model updates to capture the evolving trends. We plan to assess data and concept drifts for sustained model efficacy and demonstrate the CI/CD process, ensuring streamlined model updates and optimizations.

We are moving towards a future where every action and interaction of ours will be a data point, fundamentally shaping how we navigate and understand the world. Thus, our PII detection project marks a genuine leap forward in the realm of data privacy and security. Our future focus includes real-time monitoring, privacy-preserving techniques, and compliance features to swiftly respond to breaches and meet global regulations. Our core mission is to promote trust and ethical data practices in today's data-driven landscape, ensuring a safer and better world for everyone!

## Dataset Information

This Dataset is featured by The Learning Agency Lab in Kaggle. It consists of approximately 6800 essays authored by students participating in a massively open online course. Each essay was crafted in response to a unified assignment prompt, prompting students to integrate course content with real-world challenges. Supplementary data are available for this project, which may be incorporated into the analysis if additional data are needed to enhance model performance. To facilitate ongoing model refinement, the dataset will be partitioned into a stream of data points, ensuring a continuous supply of data for further training as the model's performance metrics fluctuate. Currently, the plan entails periodic retraining and data retrieval every month.

## Data Card

Shape: 6807 * 5

| Variable Name      | Data Type | Description                                                  |
| ------------------ | --------- | ------------------------------------------------------------ |
| Document           | int       | An integer identifier for the essay                          |
| full_text          | string    | A UTF-8 representation of the essay                          |
| tokens             | list      | Each word of string type stored in a list                    |
| trailing_whitespace| list      | A list of boolean values indicating whether each token is followed by whitespace |
| labels             | list      | A token label in BIO format                                  |

The labels being the target feature which has different classes to predict,

| Label Class    | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| NAME_STUDENT    | The full or partial name of a student who is not necessarily the author of the essay. This excludes instructors, authors, and other person names. |
| EMAIL           | A student's email address.                                  |
| USERNAME        | A student's username on any platform.                       |
| ID_NUM          | A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number. |
| PHONE_NUM       | A phone number associated with a student.                   |
| URL_PERSONAL    | A URL that might be used to identify a student.            |
| STREET_ADDRESS  | This holds the student's address.                           |

## Data Sources

The data source for this project has been sourced from a Kaggle competition on the PII data detection challenge.

Source: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data

## Installation

This project requires Python >= 3.8. Please make sure that you have the correct Python version installed on your device. Additionally, this project is compatible with Windows, Linux, and Mac operating systems.

### Prerequisites

- git
- python>=3.8
- docker daemon/desktop should be running

### User Installation

The User Installation Steps are as follows:

1. Clone the git repository onto your local machine:
   ```
   git clone https://github.com/rayapudisaiakhil/PII-Data
   ```
2. Check if python version >= 3.8 using this command:
   ```
   python --version
   ```
3. Check if you have enough memory
  ```docker
  docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
  ```
**If you get the following error, please increase the allocation memory for docker.**
  ```
  Error: Task exited with return code -9 or zombie job
  ```
4. After cloning the git onto your local directory, please edit the `docker-compose.yaml` with the following changes:

  ```yaml
  user: "1000:0" # This is already present in the yaml file but if you get any error regarding the denied permissions feel free to edit this according to your uid and gid
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com # If you are using other than gmail to send/receive alerts change this according to the email provider.
  AIRFLOW__SMTP__SMTP_USER: # Enter your email 'don't put in quotes'
  AIRFLOW__SMTP__SMTP_PASSWORD: # Enter your password here generated from google in app password
  AIRFLOW__SMTP__SMTP_MAIL_FROM:  # Enter your email
 - ${AIRFLOW_PROJ_DIR:-.}/dags: #locate your dags folder path here (eg:/home/vineshgvk/PII-Data-1/dags)
 - ${AIRFLOW_PROJ_DIR:-.}/logs: #locate your project working directory folder path here (eg:/home/vineshgvk/PII-Data-1/logs)
 - ${AIRFLOW_PROJ_DIR:-.}/config: #locate the config file from airflow (eg:/home/vineshgvk/airflow/config)
  ```
5. In the cloned directory, navigate to the config directory under PII-Data and place your key.json file from the GCP service account for handling pulling the data from GCP.
6. Run the Docker composer.
   ```
   docker compose up
   ```
7. To view Airflow dags on the web server, visit https://localhost:8080 and log in with credentials
   ```
   user: airflow2
   password: airflow2
   ```
8. Run the DAG by clicking on the play button on the right side of the window
9. Stop docker containers (hit Ctrl + C in the terminal)
    
# Tools Used for MLOps

- GitHub Actions
- Docker
- Airflow
- TensorFlow Data Validation (TFDV)
- Data Version Control (DVC)
- Google Cloud Platform (GCP)
- ML flow
- StreamLit

## GitHub Actions

GitHub Actions is configured to initiate workflows upon pushes and pull requests to any branch, including the "Name**" and main branches.

When a new commit is pushed, the workflow triggers a build process `pytest` and `pylint`. This process produces test reports in XML format, which are then stored as artifacts. The workflow is designed to locate and execute test cases situated within the test directory that correspond to modules in the dags directory. Additionally, by utilizing `pylint`, the workflow assesses the code for readability, potential security issues, and adequate documentation. Upon the successful completion of these build checks , feature branches are merged into the main branch.

## Docker and Airflow

The `docker-compose.yaml` file contains the code neccessary to run Airflow. Through the use of Docker and containerization, we are able to ship our datapipeline with the required dependencies installed. This makes it platform indepedent, whether it is windows, mac or linux, our data pipeline should run smooth.

## TensorFlow Data Validation (TFDV)

TFDV stands for TensorFlow Data Validation. TFDV provides functionalities for data analysis, schema inference, and data validation, allowing users to ensure that their data meets the expectations of their ML models. We used TFDV to know the statsgen and schema of our incoming data.

## Data Version Control (DVC)

DVC (Data Version Control) is an open-source tool essential for data versioning in machine learning projects. It tracks changes in datasets over time, ensuring reproducibility and traceability of experiments. By storing meta-information separately from data, DVC keeps Git repositories clean and lightweight. It integrates seamlessly with Git, allowing for efficient management of code, data, and models. This dual-repository approach simplifies collaboration and ensures that project states can be recreated easily. DVC's focus on data versioning is critical for maintaining the integrity and reliability of machine learning workflows.

## Google Cloud Platform (GCP)

We used GCP for storing the data pulled from Kaggle API. Our data version control is managed, hosted, and tracked on the Google Cloud Platform, which effortlessly accommodates large datasets and their versioning for building strong ETL pipelines. It enables simultaneous access and modification of data by multiple users, with built-in versioning features allowing easy retrieval of previous versions.

GCP facilitated efficient ETL implementation, preserving intermediate files across various modular tasks. One must set up a service account to use Google Cloud Platform services using below steps. 

1. Go to the GCP Console: Visit the Google Cloud Console at https://console.cloud.google.com/.

2. Navigate to IAM & Admin > Service accounts: In the left-hand menu, click on "IAM & Admin" and then select "Service accounts".

3. Create a service account: Click on the "Create Service Account" button and follow the prompts. Give your service account a name and description.

4. Assign permissions: Assign the necessary permissions to your service account based on your requirements. You can either grant predefined roles or create custom roles.

5. Generate a key: After creating the service account, click on it from the list of service accounts. Then, navigate to the "Keys" tab. Click on the "Add key" dropdown and select "Create new key". Choose the key type (JSON is recommended) and click "Create". This will download the key file to your local machine.

**You can avoid these steps of creating a GCP bucket, instead you could raise a request to access our GCP bucket**

![image](images/dcv_gcp_bucket.png)

<hr>


# Overall ML Project PipeLine

![image](images/airflow_dag.png)



## Pipeline Optimization

![image](images/airflow_gantt_chart.png)

**Gantt Chart**: It is a popular project management tool used to visualize and track the progress of tasks or activities over time. It provides a graphical representation of a pipeline's schedule, showing when each task is planned to start and finish.

* We have minimized the execution time by running `anomaly_detect.py` in parallel with `missing_values_removal.py` after observing the initial Gantt chart.

## Data Pipeline

In this project, the data pipeline is made up of various linked modules, each tasked with processing the data through specific operations. We employ Airflow and Docker to manage and encapsulate these modules, treating each one as a distinct task within the primary data pipeline DAG (data pipeline).

We utilize Apache Airflow for our pipeline. We create a DAG with our modules.

# Data Pipeline Components
![Data Pipeline Components](https://github.com/rayapudisaiakhil/PII-Data/raw/main/images/Data%20Pipeline.png)

## 1. Downloading Data
In the initial phase, the dataset is fetched and extracted into the designated data folder using the following modules:

- **data_slicing_batches_task.py**: This script ensures downloading and extracting the dataset `train.json` from Google Cloud bucket into the `dags/processed` location in your folder structure.After extracting the data, we proceed with data slicing.

## 2. Anomaly Detection
Prior to moving to inference data and model training, it is very important to detect any anomalies present in the data. The script `anomalyDetect.py` performs anomaly detection and verifies data integrity and quality in `train.json` from GCP.

- **anomalyDetect.py**: Specifically, this script performs several key checks like text length validation, sample size check, data type validation, token length check, trailing whitespace check, and label validation on the `train.json` file loaded from GCP bucket.Alerts will be sent when Anomaly’s are detected.

## 3. Cleaning Data
Data quality is extremely important in machine learning for reliable results. Hence, we conduct exploratory data analysis to ensure high-quality data, which is essential for effective model training and decision-making.

The components involved in this process are:

- **missing_values.py**: Removes missing values from `train.json` and saves the cleaned data frame as `missing_values.pkl` into the `dags/processed/` folder.

- **duplicates.py**: Identifies and eliminates duplicate records to preserve data integrity and then saves the data frame as `duplicate_removal.pkl` into the `dags/processed/` folder.

- **resampling.py**: Addresses class imbalance by downsampling and saves the data as `resampled.json` into the `dags/processed/` folder.

- **label_encoder.py**: Processes label data from `resampled.json` by mapping labels to numeric IDs and saves data as `label_encoder_data.json` into the `dags/processed/` folder.

- **tokenize_data.py**: Tokenizes text data from `label_encoder_data.json`, adds input ids and offset mapping, then saves `tokenized_data.pkl` into the `dags/processed/` folder for model building use.

Each module within the pipeline retrieves data from an input pickle path, performs processing operations, and saves the processed data to an output pickle path. The seamless integration of these modules within Airflow facilitates a structured and optimized data processing workflow.

## 4. Stats Gen
It is very important to look at data and understand it from the feature level. This helps us avoid any potential discrepancies and biases in our model results. This is where Stats Gen (Statistical Data Generation) comes into the picture.

Stats Gen is used to understand data distributions, detect patterns, and make informed decisions about model selection and evaluation. These statistical insights guide feature engineering and help assess model performance, leading to improved predictive accuracy and generalization capabilities.

Using TFDV in the `Statsgen.ipynb` notebook, we are able to extract useful statistics and schema from the input file `train.json`.

The `Statsgen.ipynb` notebook generates the following outputs:
- A TFRecord file containing the converted JSON data.
- Statistics about the data, including the number of examples, feature types, and unique values.
- A visualization of the statistics, showing the distribution of values for each feature.
- A schema that describes the structure of the data.
This information can be used to improve the quality of the data and train machine learning models more effectively.

![image](images/statsgen.png)

![image](images/image.png)



## 5. Email Alerts

We set up email alerts by configuring SMTP settings in `docker-compose.yaml` (refer to step 4 in user installation above) to receive instant alerts as email notifications upon task or DAG (Directed Acyclic Graph) failures and successes. These email alerts are crucial for monitoring and promptly addressing issues in data pipelines, ensuring data integrity, and minimizing downtime.

<hr>

# Machine Learning Modeling Pipeline
We've set up our machine learning pipeline on Google Cloud Platform (GCP). We uploaded our codebase and created Docker images. After that, we uploaded the Docker images to the Artifact Registry. We then used  (what did we use for deployment )for training and deploying our model.

# Machine Learning Pipeline Components

## 1. Trainer

Trainer Components:

Dockerfile: Utilized to execute the training job.

train.py: This Python script constructs the model using training data sourced from Google Cloud and subsequently saves it to Local Environment.

file.py: Contains the X algorithm, functions for outlier removal, and hyperparameter tuning processes.

## 2. Serve

The components are designed to deploy the ML model on where are we deploying following its training:

predict.py: 

Dockerfile: Employed to host the serving module, facilitating the deployment of the model on Vertex AI.

## 3. Model Pipeline

build.py: This script is responsible for initiating a training job by utilizing the Docker images prepared by the trainer component. After training, it manages the deployment of the trained model to an endpoint on Vertex AI, where the model will be served.

## 4. Inference

inference.py: This script is designed to send JSON input data to the model in order to obtain predictions. It handles the communication between the input data and the deployed model, facilitating the inference process.

# Experimental tracking pipeline (MLFLOW)

For monitoring our experimental machine learning pipeline, we employ MLflow, Docker, and Python. We selected three key metrics to determine the optimal model parameters from the plot provided:

Pictured: Parallel Plot for visualizing the parameter-metrics combinations for our model

# Staging, Production and Archived models (MLFLOW)

We use ML flow to manage our models across different stages—Archiving, Staging, and Production—because it enables us to leverage the models stored in the artifact registry and deploy them dynamically on a predefined port. This setup enhances our ability to reuse and serve models efficiently and flexibly.

Pictured: The image shows logs in ML flow for various experimental models, including details on parameters, metrics, and versions.

# Model Pipeline

## Train the model

The model is trained using  function. It takes x inputs and gives y outputs. The inputs are . The outputs are .

Save the model

The model is saved locally using save_and_upload_model function and uploaded to GCS.

## Hyper Parameter Tuning

This model has three hyper-parameters namely Learning Rate, Number of Training Epochs, Per Device Train Batch Size.We used ML flow to track different training runs by logging hyper parameters and performance metrics such as F1 score, precision, and recall.

Additionally we used also TensorBoard is used to visualize training metrics like loss, F1 score, precision, and recall in real-time.This visualization aids in optimizing the training process and diagnosing any issues quickly.

## Model Analysis

The model is analysed by - what function did we use to analyze the model?

Model Efficacy Report and Visuals

The plot above shows the silhouette score plots for different number of clusters. The closer it is to +1, the better it is - Add Plot


The model has the following metrics: Silhouette Score, Calinski Harabasz score and Davies Bouldin score. Below are the visuals of clusters formed after PCA and the distribution of customers into clusters.

Add Plots

# Deployment Pipeline

We've deployed the ML Model on a Vertex-AI Endpoint, utilizing StreamLit to handle requests. We've set up Model and Traffic Monitoring using X, which is linked to a Looker Dashboard to assess latency related to server load. Additionally, we use X to detect any data drifts.

# Model Insights

Insert Visualizations of our Model Insights

# Monitoring

Are we creating a Monitoring Dashboard? We've set up a Monitoring Dashboard to track the extend of data or concept drift ( if any exists ). We use ELK to log the feature input values, the predicted cluster, and timestamps(what do we log). We also record key metrics such as the (what are our key metrics )latency between predictions.

You can view the monitoring dashboard on Looker.( Add Link to the Dashboard )

# Cost Analysis:

The following is the breakdown of costs associated with the Machine Learning pipeline on Google Cloud Platform (GCP) hosted on US East1 Region.

Initial Cost Analysis

Model Training using Vertex AI: $

Deploying Model: $

Total Training and Deployment Cost: $

Serving Analysis

Daily Online Prediction for Model Serving: $

Weekly serving cost: $

Monthly serving cost: $

Yearly serving cost: $

# Contributing / Development Guide

**This is the user guide for developers**

Before developing our code, we should install the required dependencies
```python
pip install -r requirements.txt


```

## Testing
Before you push your code to GitHub, it's crucial to ensure that it meets our quality standards, including formatting, security, and functionality. To facilitate this, we recommend the following steps using `pytest` and `pylint`. These tools help identify formatting issues, potential vulnerabilities, and ensure that your test suites pass.

## Step 1: Install Required Tools

Ensure you have `pytest` and `pytest-pylint` installed in your development environment. You can install them using pip if you haven't already:
  ```
  pip install pytest 
  pip install pytest-pylint
  ```
## Step 2: Check Code Quality and Vulnerabilities

Run `pytest` with the `--pylint` option to check your code for any formatting issues or potential vulnerabilities. This step helps in identifying areas of the code that may not adhere to standard Python coding practices or might have security implications.
```
(dev environment) Project_DIR % pytest --pylint
```
Address any issues or warnings highlighted by `pylint` to improve your code's quality and maintainability.

## Step 3: Run Test Suites

To verify the functionality of your code and ensure the pipeline functionality is working as expected without any new issues, execute the test suites associated with your modules. You can run all tests in your project by simply executing pytest:
```
(dev environment) Project_DIR % pytest
```
If you prefer to run tests for a specific module, specify the path to the test file:
```
(dev environment) Project_DIR % pytest path/to/test_file.py
```
This approach allows you to isolate and debug any failures in specific areas of your project without running the entire suite of tests.

## Testing @Lahari B

Prior to uploading code to GitHub, execute the commands below on your local machine to verify a successful build.

For checking formatting issues and potential vulnerabilities in the code, use:
```
pytest --pylint
```
To execute the test suites associated with your modules, enter:
```
pytest
```
## Airflow Dags

Once your code for data pipeline modules is built successfully, copy them to dags/src/. Create your Python Operator in airflow.py within dags/src/. Set pipeline dependencies using the >> operator.

After this step, we then proceed to edit our docker-compose.yaml file

Install and set up a docker desktop for building custom images from docker-compose.yaml file.

## Docker
Additional: If your code has extra dependencies, modify the docker-compose.yaml file. Add them under the Environment section or as follows:

```
Add code here
```
Add your packages to _PIP_ADDITIONAL_REQUIREMENTS: in the docker-compose.yaml file.

Next, initialize the Airflow database as outlined in User Installation Step n. Then, continue with DAG development up to Step n.

If correctly set up, your module should appear in the DAG. If there are errors, you can check the logs and debug as needed.

## DVC Versioning

Setting up Data Versioning Control through dvc library installed as part of requirements.

1. Initialize dvc in the parent directory if not done already.
    ```python
    dvc init
    ```
2. Set up remote Google Cloud Storage connection.
    ```python
    dvc remote add -d myremote gs://<bucket>/<path>
    ```
3. Modify Google Cloud credentials to myremote by adding keys to the credential path.
    ```python
    dvc remote modify myremote credentialpath <GOOGLE-KEY-JSON-PATH>
    ```

## MLFlow

Most important declarations in the code:

1. Establish the tracking URL for MLFlow:
```
mlflow.set_tracking_uri("ADD URL")
```
2. Setting the minimum logging level to record only warnings and more severe issues (errors and critical alerts):
```
logging.basicConfig(level=logging.WARN)
```
3. Set up the logger:
```
logger = logging.getLogger(__name__)
```

4. Optionally, you may or may not choose to ignore warnings:
```
warnings.filterwarnings("ignore")
```


  
