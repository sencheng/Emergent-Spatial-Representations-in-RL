"""
Generate networks and save them as json.
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.layers import Reshape, Conv2D, Dropout, SimpleRNN, LSTM, TimeDistributed
from keras.layers import Add, Multiply
from keras.optimizers import Adam,SGD
import numpy as np
from keras.utils.vis_utils import plot_model
import keras.backend as K
import json

def calc_convolutions(input_size, kernel, stride, padding=0) :
    output = np.zeros(2)
    output[0] = np.floor(((input_size[0] - kernel[0] + 2 * padding)/stride[0]) + 1)
    output[1] = np.floor(((input_size[1] - kernel[1] + 2 * padding)/stride[1]) + 1)   
    
    return output

def CNN_original_with_dropout(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    cnn = Sequential()
    cnn.add(Reshape(input_shape,input_shape=INPUT_SHAPE, name = 'reshape_1'))
    cnn.add(Conv2D(32, (5,5), strides=(1,1)))
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(64, (4, 4), strides=(2,2)))
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(64, (3, 3), strides=(2,2)))
    cnn.add(Activation('relu'))
    cnn.add(Flatten())
    cnn.add(Dense(512))
    cnn.add(Dense(6))
    model_json = cnn.to_json()
    cnn.summary()
    with open("../experiments/networks/cnn_basic.json", "w") as json_file:
        json_file.write(model_json)
        
def deep_recurrent_network(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    model = Sequential()
    model.add(Reshape(input_shape,input_shape=INPUT_SHAPE, name = 'reshape_1'))
    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Reshape((1,1024)))
    model.add(SimpleRNN(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model_json = model.to_json()
    model.summary()
    with open("../experiments/networks/deep_recurrent.json", "w") as json_file:
        json_file.write(model_json)
        
def mixed_network(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    image_input = Input(shape = INPUT_SHAPE, name = 'image_input')
    mem_input  = Input(shape = (1,1,2,), name = 'mem_input')
    
    cnn = Reshape(input_shape, input_shape=INPUT_SHAPE, name = 'reshape_1')(image_input)
    cnn = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_1')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_2')(cnn)
    cnn = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_3')(cnn)
    cnn = Flatten()(cnn)
    cnn = Model(inputs = image_input, outputs = cnn)
    
    mlp = Dense(5, activation="relu", name='mlp_1')(mem_input)
    mlp = Flatten()(mlp)
    mlp = Model(inputs = mem_input, outputs = mlp)
    
    joint = Concatenate()([cnn.output, mlp.output])
    joint = Reshape((1,709), name = 'reshape_2')(joint)
    
    combined = SimpleRNN(25, activation="relu", name='simple_rnn')(joint)
    combined = Dense(12, activation ="linear",name='output_0')(combined)
    
    model = Model(inputs=[cnn.input,mlp.input],outputs=combined)
    
    model_json = model.to_json()
    model.summary()
    
    with open("mixed_network.json", "w") as json_file :
        json_file.write(model_json)

def deep_recurrent_network_2(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    model = Sequential()
    model.add(Reshape(input_shape,input_shape=INPUT_SHAPE, name = 'reshape_1'))
    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Reshape((1,1024)))
    model.add(SimpleRNN(12, activation='relu'))
    model.add(Dense(6))
    model_json = model.to_json()
    model.summary()
    with open("../experiments/networks/deep_recurrent_2.json", "w") as json_file:
        json_file.write(model_json)
        
def deep_recurrent_network_3(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    model = Sequential()
    model.add(Reshape(input_shape,input_shape=INPUT_SHAPE, name = 'reshape_12'))
    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Reshape((1,704)))
    model.add(SimpleRNN(50, activation='relu'))
    #model.add(Reshape((1, 10)))
    #model.add(SimpleRNN(10, activation='relu'))
    model.add(Dense(4))
    model_json = model.to_json()
    model.summary()
    plot_model(model, to_file='model_plot_net.png', show_shapes=True, show_layer_names=True)
    with open("deep_recurrent_3_20", "w") as json_file:
        json_file.write(model_json)
        
def fc_network_2(input_shape, filename) : 
    INPUT_SHAPE = (1,) + input_shape
    model = Sequential()
    model.add(Reshape(input_shape,input_shape=INPUT_SHAPE, name = 'reshape_1'))
    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', name = 'analysis'))
    model.add(Dropout(0.35))
    model.add(Dense(12))
    model_json = model.to_json()
    model.summary()
    with open(filename, "w") as json_file:
        json_file.write(model_json)

def mixed_network_2(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    image_input = Input(shape = INPUT_SHAPE, name = 'image_input')
    odor_input  = Input(shape = (1,2,), name = 'odor_input')
    
    cnn = Reshape(input_shape, input_shape=INPUT_SHAPE, name = 'reshape_1')(image_input)
    cnn = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_1')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_2')(cnn)
    cnn = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_3')(cnn)
    cnn = Flatten()(cnn)
    cnn = Model(inputs = image_input, outputs = cnn)
    
    mlp = Dense(5, activation="relu", name='mlp_1')(odor_input)
    mlp = Flatten()(mlp)
    mlp = Model(inputs = odor_input, outputs = mlp)
    
    joint = Concatenate()([cnn.output, mlp.output])
    #joint = Reshape((1,709), name = 'reshape_2')(joint)
    
    combined = Dense(50, activation="relu", name='analysis')(joint)
    combined = Dropout(0.3)(combined)
    combined = Dense(12, activation ="linear",name='output_0')(combined)
    
    model = Model(inputs=[cnn.input,mlp.input],outputs=combined)
    
    model_json = model.to_json()
    model.summary()
    
    with open("mixed_network.json", "w") as json_file :
        json_file.write(model_json)
        
        
def mixed_network_3(input_shape) : 
    INPUT_SHAPE = (1,) + input_shape
    image_input = Input(shape = INPUT_SHAPE, name = 'image_input')
    mem_input  = Input(shape = INPUT_SHAPE, name = 'mem_input')
    
    cnn = Reshape(input_shape, input_shape=INPUT_SHAPE, name = 'reshape_1')(image_input)
    cnn = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_1')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_2')(cnn)
    cnn = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_3')(cnn)
    cnn = Flatten()(cnn)
    cnn = Model(inputs = image_input, outputs = cnn)
    
    cnn2 = Reshape(input_shape, input_shape=INPUT_SHAPE, name = 'reshape_10')(mem_input)
    cnn2 = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_10')(cnn2)
    cnn2 = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_20')(cnn2)
    cnn2 = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_30')(cnn2)
    cnn2 = Flatten()(cnn2)
    cnn2 = Model(inputs = mem_input, outputs = cnn2)
    
    joint = Concatenate()([cnn.output, cnn2.output])
    joint = Reshape((1,1408), name = 'reshape_2')(joint)
    
    combined = SimpleRNN(25, activation="relu", name='simple_rnn')(joint)
    combined = Dense(12, activation ="linear",name='output_0')(combined)
    
    model = Model(inputs=[cnn.input,cnn2.input],outputs=combined)
    
    model_json = model.to_json()
    model.summary()
    
    with open("mixed_network_3.json", "w") as json_file :
        json_file.write(model_json)
        
def network_with_noise(input_shape,filename) :
    image_input = Input(shape=(1,)+input_shape, name='image_input')
    noise_input = Input(shape=(1,50), name='noise_input')
    
    
    cnn = Reshape(input_shape, input_shape=(1,)+input_shape, name = 'reshape_1')(image_input)
    cnn = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_1')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_2')(cnn)
    cnn = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_3')(cnn)
    cnn = Flatten()(cnn)
    #cnn = Reshape((1,640),name='reshape_2')(cnn)
    #rnn = SimpleRNN(35, activation="relu", name='simple_rnn')(cnn)
    fc   = Dense(50, activation='relu')(cnn)
    
    noise_2 = Reshape((50,),name='reshape_3')(noise_input)
    
    fc_with_noise = Add()([noise_2,fc])
    fc_with_noise  = Dropout(0.35)(fc_with_noise)
    #rnn_with_noise = Multiply()([noise_2, rnn])
    output_layer = Dense(3)(fc_with_noise)
    
    model = Model(inputs=[image_input, noise_input], outputs=output_layer)
    model_json = model.to_json()
    model.summary()
    with open(filename, "w") as json_file :
        json_file.write(model_json)

def mixed_network_4(input_shape, filename) : 
    INPUT_SHAPE = (1,) + input_shape
    image_input = Input(shape = INPUT_SHAPE, name = 'image_input')
    vector_input  = Input(shape = (1,4,), name = 'vector_input')
    
    cnn = Reshape(input_shape, input_shape=INPUT_SHAPE, name = 'reshape_1')(image_input)
    cnn = Conv2D(32, (5,5), strides=(1,1), activation="relu", name = 'cnn_1')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2), activation="relu", name = 'cnn_2')(cnn)
    cnn = Conv2D(64, (3,3), strides=(2,2), activation="relu", name = 'cnn_3')(cnn)
    cnn = Flatten()(cnn)
    cnn = Model(inputs = image_input, outputs = cnn)
    
    mlp = Dense(50, activation="relu", name='mlp_1')(vector_input)
    mlp = Dense(25, activation='relu', name='mlp_2')(mlp)
    mlp = Flatten()(mlp)
    mlp = Model(inputs = vector_input, outputs = mlp)
    mlp = Flatten()(vector_input)
    mlp = Model(inputs = vector_input, outputs = mlp)
    
    joint = Concatenate()([cnn.output, mlp.output])
    
    combined = Dense(50, activation="relu", name='analysis')(joint)
    combined = Dropout(0.3)(combined)
    combined = Dense(3, activation ="linear",name='output_0')(combined)
    
    model = Model(inputs=[cnn.input,mlp.input],outputs=combined)
    
    model_json = model.to_json()
    model.summary()
    with open(filename, "w") as json_file :
        json_file.write(model_json)
        
        
def prettyjson(filename) : 
    with open(filename, 'r+') as f:
        data = json.load(f)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        
if __name__=='__main__' :
    filename = "egocentric_network_noise.json"
    network_with_noise((12,48,3),filename)
    prettyjson(filename)