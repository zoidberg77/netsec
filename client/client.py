import pandas as pd

import requests
import pickle
import base64

data1 = pd.read_csv('../malicious-connection-dataset/df_bad.csv').head(10)
data = pd.concat([data1])
pickled = pickle.dumps(data)
pickled_b64 = base64.b64encode(pickled)
hug_pickled_str = pickled_b64.decode('utf-8')
result = requests.request(method='get', url='http://localhost:5000/predict', data=hug_pickled_str)
print(result.content)