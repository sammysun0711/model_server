# Custom node for super resolution pipeline with libvips

This custom node takes multiple images (RGB, NHWC) with dynamic shape (width, height) as an input. It performance multiple operations to produce two output:
- resize to desired width and height
- provides `SimpleResize` and `Composition` modes

The [image-super-resolution](https://arxiv.org/abs/1807.06779) model use in this repo is download from [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo). Please find detailed model info here: [single-image-super-resolution-1032](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/single-image-super-resolution-1032).

## Build model server with custom node for super resolution pipeline
```bash
git clone https://github.com/sammysun0711/model_server.git -b super_resolution_demo
cd model_server
IMAGE_TAG_SUFFIX=-sr make docker_build
```

## Copy compiled custom nodes lib to models directory
```bash
cp src/custom_nodes/lib/ubuntu/libcustom_node_super_resolution_nhwc.so  models/
```

## Setup python environment
```bash
cd models
pip install -r requirement.txt 
```

## (Optional) Modify super resolution OpenVINO IR model input/output with OpenVINO PrePostProcessor feature
```bash
python model_preprocess.py
```

## Start model server with docker binding with 8 cores
```bash
docker run --cpuset-cpus 0-7 --rm -v ${PWD}:/models -p 9001:9001 openvino/model_server:latest-sr \
--config_path /models/image_super_resolution_config_nhwc.json --port 9001
```

## Run client with command line
```bash
python ovms_client_nhwc.py --grpc_port 9001  --input_images_dir images --model_name super_resolution \
--height 270 --width 480 --batch_size 1 -niter 100
```
