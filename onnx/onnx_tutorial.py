import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime

model_url = "".join(
    [
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ]
)
model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
# now you have super_resolution.onnx on disk
onnx_model = onnx.load(model_path)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]
print(x)

######################################################################
# Compile the model with relay
# ---------------------------------------------
target = "llvm -mcpu=broadwell"

input_name = "1"
shape_dict = {input_name: x.shape}
print(x.shape)
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print(mod)

with tvm.transform.PassContext(opt_level=1):
#    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)
    lib = relay.build(mod, target=target, params=params)

ctx = tvm.context(str(target), 0)
module = graph_runtime.GraphModule(lib["default"](ctx))
module.set_input("1", x)


######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
#tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
module.run()
output_shape=(1,1,672,672)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).asnumpy()

# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck. The luminance channel, `Y` is the output
# from the model. The chroma channels `Cb` and `Cr` are resized to match with a simple
# bicubic algorithm. The image is then recombined and converted back to `RGB`.
from matplotlib import pyplot as plt

out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
canvas = np.full((672, 672 * 2, 3), 255)
canvas[0:224, 0:224, :] = np.asarray(img)
canvas[:, 672:, :] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()

######################################################################
# Tune the model
# ---------------------------------------------
tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params, target)

dtype = "float32"
batch_size = 1
log_file = "%s-B%d-%s.json" % ("superres", batch_size, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

#run_tuning()


######################################################################
# Compile the model
# ---------------------------------------------
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
        print(type(lib))

ctx = tvm.context(str(target), 0)
module = graph_runtime.GraphModule(lib["default"](ctx))
data_tvm = tvm.nd.array((np.random.uniform(size=x.shape)).astype(dtype))
module.set_input("1", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

