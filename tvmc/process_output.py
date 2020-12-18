import os.path
import numpy as np
import mxnet as mx

from scipy.special import softmax

from tvm.contrib.download import download_testdata

with open("synset.txt", "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        a = np.argsort(scores)[::-1]

        for i in a[0:5]:
            print("class='%s' with probability=%f" % (labels[i], scores[i]))
