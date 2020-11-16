import mxnet as mx
import numpy as np
from collections import namedtuple
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import os
import time

#mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/synset.txt')
#mx.test_utils.download('https://s3.amazonaws.com/model-server/inputs/kitten.jpg')
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

Batch = namedtuple('Batch', ['data'])
def get_image(path, show=False):
    img = mx.image.imread(path)
    if img is None:
        return None
    return img

def preprocess(img):   
    transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img

def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    mod.forward(Batch([img]))
    # Take softmax to generate probabilities
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    for i in a[0:5]:
        print('class=%s ; probability=%f' %(labels[i],scores[i]))

ft = time.time()
ctx = mx.cpu()

# Load module
model_path= 'resnet50-v2-7.onnx'
sym, arg_params, aux_params = import_model(model_path)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)


# Enter path to the inference image below
img_path = "kitten.jpg"
total_time = 0
runs = 1
run_times = []
for i in range(runs):

    t = time.time() 
    predict(img_path)
    elapsed = time.time() - t
    run_times.append(elapsed)

print("onnx average run time: %s" % (np.average(run_times)))