from data.save_records import main
from data.dataset import build_tfrecord_dataset
import requests
import json
import numpy as np

main()

"""
data_train, data_eval = build_tfrecord_dataset('./data/records/', 1, 1)
xs, ys = next(iter(data_train))
xs_lists = []
for x in xs:
    xs_lists.append(np.squeeze(x.numpy()).flatten().tolist())

headers = {"content-type": "application/json"}
# Input Tensors in row ("instances") or columnar ("inputs") format.
req_data = {
    "signature_name": "predict",
    'instances': xs_lists
}

ret = requests.get('http://localhost:20200/v1/models/saved_model.pb', headers=headers)
print(ret.text)

xs_str = json.dumps(req_data)
print('fucking', xs_str)
ret = requests.post('http://localhost:20200/v1/models/saved_model.pb:predict', data=xs_str, headers=headers)
print(ret.text)
"""
