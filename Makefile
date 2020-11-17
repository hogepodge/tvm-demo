demo: onnx.txt tvm.txt tvm-autotuned.txt
	cat onnx.txt tvm.txt tvm-autotuned.txt

demo-cache: onnx.txt tvm.txt tvm-autotuned-cache.txt
	cat onnx.txt tvm.txt tvm-autotuned-cache.txt

cache: autotuner_records-cache.json resnet50-v2-7-autotuned-cache.tvm

autotuner_records-cache.json:
	cp autotuner_records.json autotuner_records-cache.json

clean:
	rm -f autotuner_records.json \
	imagenet-simple-labels.json \
	kitten.jpg \
	kitten.npz \
	onnx.txt \
	predictions.npz \
	resnet50-v2-7-autotuned.tvm \
	resnet50-v2-7-autotuned-cache.tvm \
	resnet50-v2-7.onnx \
	resnet50-v2-7.tvm \
	synset.txt \
	tvm-autotuned.txt \
	tvm-autotuned-cache.txt \
	tvm.txt \

cacheclean: clean
	rm autotuner_records-cache.json

resnet50-v2-7.onnx:
	$(info ---===*** Downloading resnet50-v2-7.onnx ***===---)
	curl -L https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx -o resnet50-v2-7.onnx

kitten.jpg:
	$(info ---===*** Downloading kitten.jpg ***===---)
	curl -L https://s3.amazonaws.com/model-server/inputs/kitten.jpg -o kitten.jpg

kitten.npz: kitten.jpg
	$(info ---===*** Converting kitten.jpg to kitten.npz ***===---)
	python3 process_input.py

synset.txt:
	$(info ---===*** Downloading additional onnx model file synset.txt ***===---)
	curl -L https://s3.amazonaws.com/onnx-model-zoo/synset.txt -o synset.txt

imagenet-simple-labels.json:
	$(info ---===*** Downloading additional labels for onnx model ***===---)
	curl -L https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json -o imagenet-simple-labels.json

onnx.txt: kitten.npz synset.txt imagenet-simple-labels.json resnet50-v2-7.onnx
	$(info ---===*** Benchmarking onnx ***===---)
	echo "=== Begin onnx benchmark results ===" > onnx.txt
	python3 onnxbenchmark.py >> onnx.txt
	echo "=== End onnx benchmark results ===" >> onnx.txt
	cat onnx.txt

resnet50-v2-7.tvm: resnet50-v2-7.onnx
	$(info ---===*** Compiling unoptimized TVM model for resnet50-v2-7 ***===---)
	tvmc compile \
	--target "llvm -mcpu=broadwell" \
	--output resnet50-v2-7.tvm \
	resnet50-v2-7.onnx 2> /dev/null

autotuner_records.json: resnet50-v2-7.onnx
	$(info ---===*** TVM autotuning model for resnet50-v2-7 ***===---)
	tvmc tune \
	--target "llvm -mcpu=broadwell" \
	--output autotuner_records.json \
	resnet50-v2-7.onnx 2> /dev/null

resnet50-v2-7-autotuned.tvm: resnet50-v2-7.onnx autotuner_records.json
	$(info ---===*** Compiling optimized TVM model for resnet50-v2-7 ***===---)
	tvmc compile \
	--target "llvm -mcpu=broadwell" \
	--tuning-records autotuner_records.json  \
	--output resnet50-v2-7-autotuned.tvm \
	resnet50-v2-7.onnx 2> /dev/null

resnet50-v2-7-autotuned-cache.tvm: resnet50-v2-7.onnx
	$(info ---===*** Compiling optimized TVM model for resnet50-v2-7 from cache ***===---)
	tvmc compile \
	--target "llvm -mcpu=broadwell" \
	--tuning-records autotuner_records-cache.json  \
	--output resnet50-v2-7-autotuned-cache.tvm \
	resnet50-v2-7.onnx 2> /dev/null

tvm.txt: kitten.npz synset.txt resnet50-v2-7.tvm
	$(info ---===*** Benchmarking unoptimized TVM model for resnet50-v2-7 ***===---)
	echo "=== Begin tvm benchmark results ===" > tvm.txt
	tvmc run \
	--inputs kitten.npz \
	--output predictions.npz \
	--print-time \
	--repeat 5 \
	resnet50-v2-7.tvm > tvm.txt
	python3 process_output.py >> tvm.txt
	echo "=== End tvm benchmark results ===" >> tvm.txt
	echo tvm.txt

tvm-autotuned.txt: kitten.npz synset.txt resnet50-v2-7-autotuned.tvm
	$(info ---===*** Benchmarking optimized TVM model for resnet50-v2-7 ***===---)
	echo "=== Begin tvm-autotuned benchmark results ===" >> tvm-autotuned.txt
	tvmc run \
	--inputs kitten.npz \
	--output predictions.npz \
	--print-time \
	--repeat 5 \
	resnet50-v2-7-autotuned-cache.tvm > tvm-autotuned.txt
	python3 process_output.py >> tvm-autotuned.txt
	echo "=== End tvm benchmark results ===" >> tvm-autotuned.txt
	echo tvm-autotuned.txt

tvm-autotuned-cache.txt: kitten.npz synset.txt resnet50-v2-7-autotuned-cache.tvm
	$(info ---===*** Benchmarking optimized TVM model for resnet50-v2-7 from cache ***===---)
	echo "=== Begin tvm-autotuned-cache benchmark results ===" > tvm-autotuned-cache.txt
	tvmc run \
	--inputs kitten.npz \
	--output predictions.npz \
	--print-time \
	--repeat 5 \
	resnet50-v2-7-autotuned-cache.tvm >> tvm-autotuned-cache.txt
	python3 process_output.py >> tvm-autotuned-cache.txt
	echo "=== End tvm benchmark results ===" >> tvm-autotuned-cache.txt
	echo tvm-autotuned-cache.txt
