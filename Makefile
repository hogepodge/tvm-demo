demo: onnx.txt tvm.txt tvm-autotuned.txt
	cat onnx.txt tvm.txt tvm-autotuned.txt

demo-cache: onnx.txt tvm.txt tvm-autotuned-cache.txt
	cat onnx.txt tvm.txt tvm-autotuned-cache.txt

cache: autotuner_records-cache.json resnet50-v2-7-autotuned-cache.tvm

autotuner_records-cache.json:
	cp autotuner_records.json autotuner_records-cache.json

resnet50-v2-7-autotuned-cache.tvm:
	cp resnet50-v2-7-autotuned.tvm resnet50-v2-7-autotuned-cache.tvm

clean:
	rm onnx.txt tvm.txt tvm-autotuned.txt

realclean: clean
	rm resnet50-v2-7.onnx

cacheclean: realclean
	rm autotuner_records-cache.json resnet50-v2-7-autotuned-cache.tvm tvm-autotuned-cache.txt

resnet50-v2-7.onnx:
	curl -L https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx -o resnet50-v2-7.onnx

kitten.jpg:
	curl -L https://s3.amazonaws.com/model-server/inputs/kitten.jpg -o kitten.jpg

kitten.npz:
	python3 process_input.py

synset.txt:
	curl -L https://s3.amazonaws.com/onnx-model-zoo/synset.txt -o synset.txt

onnx.txt: kitten.jpg synset.txt
	python3 onnxbenchmark.py > onnx.txt
	cat onnx.txt

resnet50-v2-7.tvm: resnet50-v2-7.onnx
	tvmc compile \
	--target "llvm -mcpu=broadwell" \
	--output resnet50-v2-7.tvm \
	resnet50-v2-7.onnx

autotuner_records.json: resnet50-v2-7.onnx
	tvmc tune \
	--target "llvm -mcpu=broadwell" \
	--output autotuner_records.json \
	resnet50-v2-7.onnx

resnet50-v2-7-autotuned.tvm: resnet50-v2-7.onnx autotuner_records.json
	tvmc compile \
	--target "llvm -mcpu=broadwell" \
	--tuning-records autotuner_records.json  \
	--output resnet50-v2-7-autotuned.tvm \
	resnet50-v2-7.onnx

tvm.txt: kitten.npz synset.txt resnet50-v2-7.tvm
	tvmc run \
	--inputs kitten.npz \
	--output predictions.npz \
	--print-time \
	--repeat 5 \
	resnet50-v2-7.tvm > tvm.txt
	python3 process_output.py

tvm-autotuned-cache.txt: kitten.npz synset.txt resnet50-v2-7-autotuned-cache.tvm
	tvmc run \
	--inputs kitten.npz \
	--output predictions.npz \
	--print-time \
	--repeat 5 \
	resnet50-v2-7-autotuned-cache.tvm > tvm-autotuned-cache.txt
	python3 process_output.py
