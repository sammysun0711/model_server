from openvino.runtime import Core, Layout, Type, serialize
from openvino.preprocess import ColorFormat, PrePostProcessor

core = Core()
input_tensor_name_1 = "0"
input_tensor_name_2 = "1"
output_tensor_name = "129"

model_path = "./super_resolution/1/single-image-super-resolution-1032.xml"
model = core.read_model(model_path)

ppp = PrePostProcessor(model)

#Input 1
ppp.input(input_tensor_name_1).tensor().set_element_type(Type.u8)
ppp.input(input_tensor_name_1).tensor().set_color_format(ColorFormat.RGB)
ppp.input(input_tensor_name_1).tensor().set_layout(Layout('NHWC'))
ppp.input(input_tensor_name_1).model().set_layout(Layout('NCHW'))

#Input 2
ppp.input(input_tensor_name_2).tensor().set_element_type(Type.u8)
ppp.input(input_tensor_name_2).tensor().set_color_format(ColorFormat.RGB)
ppp.input(input_tensor_name_2).tensor().set_layout(Layout('NHWC'))
ppp.input(input_tensor_name_2).model().set_layout(Layout('NCHW'))

#Output 1
#ppp.output(output_tensor_name).tensor().set_layout(Layout('NHWC'))
ppp.output(output_tensor_name).tensor().set_element_type(Type.u8)
ppp.output(output_tensor_name).tensor().set_layout(Layout('NHWC'))
ppp.output(output_tensor_name).model().set_layout(Layout('NCHW'))


# layout and precision conversion is inserted automatically,
# because tensor format != model input format
model = ppp.build()
serialize(model, './super_resolution_model_preprocessed/1/single-image-super-resolution-1032.xml', 
                 './super_resolution_model_preprocessed/1/single-image-super-resolution-1032.bin')
