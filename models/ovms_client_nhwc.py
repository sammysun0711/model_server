#
# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.append("../demos/common/python")

import argparse
import datetime
import grpc
import numpy as np
import os
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs
import pyvips

parser = argparse.ArgumentParser(description='Demo for super resolution requests via TFS gRPC API analyses input images')

parser.add_argument('--input_images_dir', required=False, help='Directory with input images', default="images")
parser.add_argument('--output_dir', required=False, help='Directory for storing images with detection results', default="results")
parser.add_argument('--batch_size', required=False, help='How many images should be grouped in one batch', default=1, type=int)
parser.add_argument('--width', required=False, help='How the input image width should be resized in pixels', default=480, type=int)
parser.add_argument('--height', required=False, help='How the input image width should be resized in pixels', default=270, type=int)
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9001, help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name',required=False, default='super_resolution', help='Specify the model name')
parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--server_cert', required=False, help='Path to server certificate')
parser.add_argument('--client_cert', required=False, help='Path to client certificate')
parser.add_argument('--client_key', required=False, help='Path to client key')
parser.add_argument('--niter', required=False, default=100, type=int, help='Number of iteration for performance evaluation')
parser.add_argument('--save_img', required=False, default=False, type=bool, help='Whether save result image')
args = vars(parser.parse_args())

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
channel = None
if args.get('tls'):
    server_ca_cert, client_key, client_cert = prepare_certs(server_cert=args['server_cert'],
                                                            client_key=args['client_key'],
                                                            client_ca=args['client_cert'])
    creds = grpc.ssl_channel_credentials(root_certificates=server_ca_cert,
                                         private_key=client_key, certificate_chain=client_cert)
    channel = grpc.secure_channel(address, creds, options = options)
else:
    channel = grpc.insecure_channel(address, options = options)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

files = os.listdir(args['input_images_dir'])
batch_size = args['batch_size']
print(files)

#imgs=np.zeros((0, 1, args['height'], args['width'], 3), np.dtype('<f'))
imgs=np.zeros((0, 1, args['height'], args['width'], 3), np.dtype(np.uint8))
file_path = os.path.join(args['input_images_dir'], files[0])
img = pyvips.Image.new_from_file(file_path, access='sequential')
print("type(img): ", type(img))
img = img.numpy()
print("img.shape: ", img.shape)
img = img.reshape(1,args['height'],args['width'], 3)
img = np.expand_dims(img, axis=0)
print("img.shape: ", img.shape)
for i in range(batch_size):
    imgs = np.append(imgs, img, axis=0) 
print("imgs shape: ", imgs.shape)
request = predict_pb2.PredictRequest()
print("request.model_spec.name: ", args['model_name'])
request.model_spec.name = args['model_name']
request.inputs["data"].CopyFrom(make_tensor_proto(imgs, shape=(imgs.shape)))
processing_times = np.zeros((0),int)
for i in range(args['niter']):
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    if args['save_img']:
        outputs = make_ndarray(result.outputs["129"])
        output = np.reshape(outputs, (1080,1920,3))
        #output*=255
        vips_output = pyvips.Image.new_from_array(output)
        vips_output.write_to_file("vips_result_image.jpeg")

print("processing time: ", processing_times)
print_statistics(processing_times, batch_size)
