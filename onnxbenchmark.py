import numpy as np 
import onnxruntime
import onnx
from onnx import numpy_helper
import json
import time

from PIL import Image, ImageDraw, ImageFont

test_data_dir = 'resnet50v2/test_data_set'
test_data_num = 3


session = onnxruntime.InferenceSession('resnet50-v2-7.onnx', None)

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

labels = load_labels('imagenet-simple-labels.json')

input_data = np.load('kitten.npz')['data']

start = time.time()
input_name = 'data'
raw_result = session.run([], {input_name: input_data})
end = time.time()
res = postprocess(raw_result)

inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

sort_idx = np.argsort(np.squeeze(res))[::-1]
print('============ Top 5 labels are: ============================')
for i in sort_idx[:5]:
    print("class='%s' with probability=%f" % (labels[i], res[i]))
print('===========================================================')

