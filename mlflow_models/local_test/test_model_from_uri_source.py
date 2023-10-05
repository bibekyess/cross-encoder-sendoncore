import mlflow

# Make sure you set the appropriate environment variable for connecting the Minio database
logged_model = 's3://mlflow/0/d3ac0e6fa8824a2980b8041c05612b66/artifacts/cross_encoder_pyfunc'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model, dst_path='./model_from_mlflow')

inputs = {
            "query": ["Query"],
            "paragraphs": ["Paragraph1", "Paragraph2", "Paragraph3"]
         }

inference_request = {
    "parameters": {
        "content_type": "pd"
    },
    'inputs': [{
        'name': 'query', 
        'shape': [1, 1], 
        'datatype': 'BYTES', 
        'parameters': {'content_type': 'str'}, 
        'data': ['Query']
        }, 
        {
        'name': 'paragraphs', 
        'shape': [3, 1], 
        'datatype': 'BYTES', 
        'parameters': {'content_type': 'str'}, 
        'data': ['Paragraph1', 'Paragraph2', 'Paragraph3']
        }]
        }

scores = loaded_model.predict(inference_request)

print(scores)