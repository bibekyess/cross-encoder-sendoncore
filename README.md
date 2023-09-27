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
Download any cross-encoder model from Hugging Face:
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
