# Cross-Encoder Seldon Core

This repository contains code and instructions for serving custom NLP models like `cross encoder` in local environment using docker or opensource serving infrastructure like SeldonCore.

## Getting Started

### Environment Setup

Create a Conda environment and activate it using the following commands:

```bash
conda create -n cross-encoder-seldoncore python=3.11.5
conda activate cross-encoder-seldoncore
```

Install the Python packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### Model Download
Download any cross-encoder model from Hugging Face. For this github example, I am using `ms-marco-MiniLM-L-6-v2`:
```bash
cd huggingface_models/mlserver_image_build/model
git lfs install
git clone git@hf.co:cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Local Serving
For running locally, navigate to the local serving directory:
```bash
cd huggingface_models/local_serving/decode_args
# or
cd huggingface_models/local_serving/str_json_dump
# or
cd huggingface_models/local_serving/str_numpy_codec
```
*Note*: The three directories, `decode_args`, `str_json_dump`, and `str_numpy_codec`, showcase three different methods of passing an input payload when creating a custom runtime in MLServer.

1. The `@decode_args` decorator attempts to automatically match the data type as defined in the input type. However, it may not always work as expected.

2. In the `str_json_dump` approach, the complete input payload is converted to a string format and then passed.

3. In the `str_numpy_codec` approach, MLServer's codecs are used to encode and decode inputs that are strings and numpy lists, respectively.

Start the MLServer:
```bash
mlserver start .
```
To test, open another terminal and execute:
```bash
python test.py
```

### Docker Image Build
To build a Docker image for deployment, navigate to the Dockerfile directory:
```bash
cd huggingface_models/mlserver_image_build
```
We can build the image using `docker build` command or `mlserver build`.
Using `docker build` command:
```bash
mlserver dockerfile . # You can customize the generated dockerfile
docker build -t <username>/<reponame>:<tag> .
```
Using `mlserver build` command:
```bash
mlserver build -t <username>/<reponame>:<tag> .
```

### Seldon Serving
Navigate to the Seldon serving directory:
```bash
cd huggingface_models/seldon_serving
```

Testing via Docker
Run the Docker container:
```bash
docker run -it --rm -p 8080:8080 <username>/<reponame>:<tag>
python test_docker.py
```

Testing via Kubernetes (with Kubeflow Gateway)
Apply the configuration files:
```bash
kubectl apply -f config
python test_kubernetes.py
```



## For MLflow models
### Prerequisites
Before getting started, ensure that the MLflow tracking server is running, and you have set up the environment for the database backend you are using. For example, if you are using the S3 artifact store, follow these steps (for local testing):
```bash
# Set environment variables for S3 artifact store
export MLFLOW_S3_ENDPOINT_URL=<YOUR-VALUE>
export AWS_ACCESS_KEY_ID=<YOUR-VALUE>
export AWS_SECRET_ACCESS_KEY=<YOUR-VALUE>
export MLFLOW_S3_IGNORE_TLS="true"
```

### Logging an MLflow Model
```bash
cd mlflow_models/log_mlflow_model
python log_model_pyfunc.py
```

### Local testing
```bash
cd mlflow_models/local_test
mkdir model_from_mlflow
python test_model_pyfunc.py
```

### Seldon Serving:Non-Reusable Model Server
```bash
cd mlflow_models/seldon_serving/non_reusable_model_server
mkdir model_from_mlflow
python download_model.py
docker build -t "bibekyess/non-reusable-model-server:v1" .
kubectl apply -f config
```

### Seldon Serving:Reusable Model Server
```bash
cd mlflow_models/seldon_serving/reusable_model_server
docker build -t "bibekyess/reusable-model-server:v1" .
kubectl apply -f config
```


In the reusable_model_server deployment, you need to consider two things:
1. Create a `requirements.txt` file, which can be obtained from inside the `model_from_mlflow` directory or customized as needed.

2. Edit the Seldon-config configmap by running:
```bash
kubectl edit configmap -n seldon-system seldon-config
```
Under `data.predictor_servers`, add the following:
```bash
"CUSTOM_IMPLEMENTATION": {
  "protocols": {
    "v2": {
      "defaultImageVersion": "v1",
      "image": "<username>/<image-name>"
    }
  }
}
```

### Testing Model Serving
```bash
cd mlflow_models/seldon_serving/
python inference.py
```

