import mlflow
from mlflow import MlflowClient
from pprint import pprint

mlflow.set_tracking_uri('http://192.168.0.29:5000') # FIXME

# Make sure you set the appropriate environment variable for connecting the Minio database
logged_model = 'models:/cross-encoder-sendoncore/Production'

client = MlflowClient()
source = ''
for mv in client.search_model_versions("name='cross-encoder-sendoncore'"):
    mv = dict(mv)
    if mv['current_stage'] == 'Production':
        source = mv['source']
    # pprint(dict(mv), indent=4)
print(source)

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