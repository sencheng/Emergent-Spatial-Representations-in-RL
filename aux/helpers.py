import numpy as np
import scipy.cluster.hierarchy as hcluster
from keras import backend as K
from keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from rl import callbacks
import json


def generate_from_grid(world, x_limits, y_limits, angle, grid_size,
                       images_file) :
    """
    Specify a grid size and environment to generate 
    input images at grid points with (x,y) position.

    Note : saves original 256x64 images from Blender.
    Remember to resize them to match your network 
    input before using.
    """
    
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])

    xx,yy = np.meshgrid(X,Y)
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T
        
    images = []
    
    for point in grid_points : 
        observation = world.stepSimNoPhysics(point[0],point[1], angle)
        images.append(observation[3])
        
    np.save(images_file, images)

    return X,Y


def generate_from_grid_angle(world, x_limits, y_limits, angle, grid_size,
                       images_file, points_file) :
    """
    Specify a grid size and environment to generate 
    input images at grid points with (x,y) position 
    while being constantly oriented towards a fixed point.

    Note : saves original 256x64 images from Blender.
    Remember to resize them to match your network 
    input before using.
    """
    def calculateAngle(center, node) : 
    
        delta_x =  center[0] - node[0] 
        delta_y =  center[1] - node[1]
        theta_radians = np.arctan2(delta_y, delta_x)
    
        return np.degrees(theta_radians)
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])

    xx,yy = np.meshgrid(X,Y)
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T
        
    images = []
    origin = np.array([0.0, 0.0])
    node_angle = 0.0 
    for point in grid_points : 
        node_angle = calculateAngle(origin, point) + angle
        observation = world.stepSimNoPhysics(point[0],point[1], node_angle)
        images.append(observation[3])
        
    np.save(images_file, images)
    np.save(points_file, grid_points)
    
    return X,Y
    
def generate_uniform_random():
    pass

def generate_from_topology():
    pass
        
def build_model(network_file) :
    """ Build a model from a json file.
    
        network_file : path to network file
    """
    
    json_file = open(network_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)

def resize_images(images_list, width, height) : 
    """ Resize all images in a list to fit network 
        input
        
    """
    
    resized_images = []
    for i in range(len(images_list)) :
        new_img = cv2.resize(images_list[i], dsize=(width,height))
        resized_images.append(new_img)
    return resized_images

def limit_field_of_view(images_list, width, view_angle) :
    """ Crop images according to viewing angle
    """
    cropped_images = []
    keep_pixels = width * view_angle /360.0
    remove_pixels = width - keep_pixels
    rm = np.ceil(remove_pixels/2).astype('int32')
    
    for i in range(len(images_list)) :
        new_img = images_list[i][:,rm:rm+np.ceil(keep_pixels).astype('int32')]
        cropped_images.append(new_img)
        
    return cropped_images
    
def get_activity_map(images, model, dense_layer_idx=9, network_from_file=True,
                     working_mem=False,mem_state=None) :
    """ Compute the spatial activity map for a neuron.
        
        images   : list of images
        points_x : list or array of x-values in the spatial map
        points_y : list or array of y-values in the spatial map
        model    : keras model (make sure your model is compiled
                                and the weights loaded before calling 
                                this function)
        dense_layer_idx : layer ID of the dense layer (or other layer
                                                       for which the spatial map
                                                       is to be computed)
        neuron_idx : the neuron in the dense layer for which the spatial
                     map is to be computed
        
    """
    if not working_mem :
        if network_from_file : 
            dense_layer_activations = K.function([model.layers[1].input],
                                                 [model.layers[dense_layer_idx].output])
            
        else :
            dense_layer_activations = K.function([model.layers[0].input],
                                                 [model.layers[dense_layer_idx].output])
        
        
        activity = []
        
        for i in range(len(images)) :
            activity.append(dense_layer_activations([[images[i]]]))
            
    else :
        if network_from_file : 
            dense_layer_activations = K.function([model.layers[1].input, model.layers[4].input],
                                                 [model.layers[dense_layer_idx].output])
            
        else :
            dense_layer_activations = K.function([model.layers[0].input],
                                                 [model.layers[dense_layer_idx].output])
        
        
        activity = []
        
        for i in range(len(images)) :
            activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,54,3),[mem_state]]))
        
        
    return activity

def get_activity_map_image_mem(images, model, dense_layer_idx=14, network_from_file=True,
                               mem_state=None) :
    """ Compute the spatial activity map for a neuron.
        
        images   : list of images
        points_x : list or array of x-values in the spatial map
        points_y : list or array of y-values in the spatial map
        model    : keras model (make sure your model is compiled
                                and the weights loaded before calling 
                                this function)
        dense_layer_idx : layer ID of the dense layer (or other layer
                                                       for which the spatial map
                                                       is to be computed)
        neuron_idx : the neuron in the dense layer for which the spatial
                     map is to be computed
        
    """


    dense_layer_activations = K.function([model.layers[1].input, model.layers[2].input],
                                         [model.layers[dense_layer_idx].output])

    activity = []
    
    for i in range(len(images)) :
        activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,54,3),
                                                 np.array([[mem_state]]).reshape(1,1,12,54,3)]))
        
        
    return activity

def get_activity_map_odor(images, model, dense_layer_idx=11, network_from_file=True,odor_signal=None) :
    """ Compute the spatial activity map for a neuron.
        
        images   : list of images
        points_x : list or array of x-values in the spatial map
        points_y : list or array of y-values in the spatial map
        model    : keras model (make sure your model is compiled
                                and the weights loaded before calling 
                                this function)
        dense_layer_idx : layer ID of the dense layer (or other layer
                                                       for which the spatial map
                                                       is to be computed)
        neuron_idx : the neuron in the dense layer for which the spatial
                     map is to be computed
        
    """

    if network_from_file : 
        dense_layer_activations = K.function([model.layers[1].input, model.layers[4].input],
                                             [model.layers[dense_layer_idx].output])
        
    else :
        dense_layer_activations = K.function([model.layers[0].input],
                                             [model.layers[dense_layer_idx].output])
    
    
    activity = []
    
    for i in range(len(images)) :
        #activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,54,3),np.array([[odor_signal[i]]])]))
        activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,48,3),np.array([[[-1,0]]])]))
        
        
    return activity

def get_activity_map_lesion(images, model, dense_layer_idx=10, network_from_file=True,
                            n_units=35, lesion_unit=None) :
    """ Compute the spatial activity map for a neuron.
        
        images   : list of images
        points_x : list or array of x-values in the spatial map
        points_y : list or array of y-values in the spatial map
        model    : keras model (make sure your model is compiled
                                and the weights loaded before calling 
                                this function)
        dense_layer_idx : layer ID of the dense layer (or other layer
                                                       for which the spatial map
                                                       is to be computed)
        neuron_idx : the neuron in the dense layer for which the spatial
                     map is to be computed
        
    """

    if network_from_file : 
        dense_layer_activations = K.function([model.layers[1].input, model.layers[6].input],
                                             [model.layers[dense_layer_idx].output])
        
    else :
        dense_layer_activations = K.function([model.layers[0].input],
                                             [model.layers[dense_layer_idx].output])
    
    
    activity = []
    noise = np.zeros(n_units)
    noise = np.ones(n_units)
    if lesion_unit is not None : 
        noise[lesion_unit] = 10.0
    print(noise)
    for i in range(len(images)) :
        #activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,54,3),np.array([[odor_signal[i]]])]))
        activity.append(dense_layer_activations([np.array([[images[i]]]).reshape(1,1,12,48,3),
                                                 np.array([[noise]])]))
        
        
    return activity

def get_activity_map_for_action(images, model, action = 0, dense_layer_idx=9,
                                network_from_file=True) :
    """ Compute the spatial activity map for a neuron.
        
        images   : list of images
        points_x : list or array of x-values in the spatial map
        points_y : list or array of y-values in the spatial map
        model    : keras model (make sure your model is compiled
                                and the weights loaded before calling 
                                this function)
        dense_layer_idx : layer ID of the dense layer (or other layer
                                                       for which the spatial map
                                                       is to be computed)
        neuron_idx : the neuron in the dense layer for which the spatial
                     map is to be computed
        
    """
    if network_from_file : 
        dense_layer_activations = K.function([model.layers[1].input],
                                             [model.layers[dense_layer_idx].output])
        
    else :
        dense_layer_activations = K.function([model.layers[0].input],
                                             [model.layers[dense_layer_idx].output])
    
    
    activity = []
    
    for i in range(len(images)) :
        q_values = model.predict([[[images[i]]]])
        if np.argmax(q_values) == action :
            print(np.argmax(q_values))
            activity.append(dense_layer_activations([[images[i]]]))
        
    return activity

def recurrent_weights_balance(agent, weights, recurrent_layer_id=10):
    agent.load_weights(weights)
    recurrent_weights = agent.model.layers[recurrent_layer_id].get_weights()
    inhibitory = 0
    excitatory = 0
    balance_matrix = np.ndarray(recurrent_weights[1].shape)
    
    for ix,iy in np.ndindex(recurrent_weights[1].shape) :
        w = recurrent_weights[1][ix,iy]
        
        if w < 0 :
            inhibitory += 1
            balance_matrix[ix,iy] = -1
        else :
            excitatory += 1
            balance_matrix[ix,iy] = +1
    
    return inhibitory, excitatory, balance_matrix

def classify_place_cell(activity_map, activity_threshold, distance_threshold) :
    #TODO : check distance from peak to where activity falls off
    indices     = np.where(activity_map < activity_threshold * np.max(activity_map))
    map_indices = np.where(activity_map > activity_threshold * np.max(activity_map))
    map_indices = np.array(map_indices).T
    activity_map[indices] = 0

    if np.count_nonzero(activity_map) <= 1 :
        return False, activity_map
    
    else :
        clusters = hcluster.fclusterdata(map_indices, distance_threshold, 
                                         criterion="distance")
        
        n_clusters = np.max(clusters)
        
        if n_clusters > 6 :
            return False, activity_map
        
        else : 
            unique, cluster_sizes = np.unique(clusters, return_counts=True)
            if np.any(cluster_sizes > 0.5 * np.size(activity_map)) :
                return False, activity_map
            elif np.all(cluster_sizes < 0.03 * np.size(activity_map)) :
                return False, activity_map
            else :
                return True, activity_map
            

def get_field_centers(place_fields) :
    field_centers = []
    for field in place_fields :
        field_centers.append(np.where(field == np.max(field)))
    return np.hstack(field_centers)


def draw_cell_mosaics(fields, fields_per_mosaic, x_limits, y_limits,
                      grid_size,folder, goal=None, thresh=0.5) :

    n_fields = np.size(fields,-1)
    n = divmod(n_fields,fields_per_mosaic)
    
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])
    
    start = 0
    for m in range(n[0]) :
        field_sliced = fields[:,:,start:start + fields_per_mosaic]
        fig, axis = plt.subplots(fields_per_mosaic,6,figsize=(12,12))
        
        
        for field in np.arange(fields_per_mosaic) :
            for angle in np.arange(6) :
        
                ax = field_sliced[:,:,field][angle]
                
                xx,yy = np.meshgrid(Y,X)
                zz = ax.reshape(len(Y), len(X))
                zz = np.where(zz < thresh * np.max(zz), 0, zz)
                zz = gaussian_filter(zz, sigma=2)
                axis[field,angle].imshow(zz,cmap='jet', origin='lower')
                if goal is not None :
                    x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                    y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                    
                    axis[field, angle].scatter([x],[y],c='white',s=100)
                axis[field,angle].axis('off')
                fig.savefig(folder+'/mosaic_%s.png'%m)
        start += fields_per_mosaic
        
    field_sliced = fields[:,:,start:start + n[1]]
    fig, axis = plt.subplots(n[1],6,figsize=(12,12))
    
    
    for field in np.arange(n[1]) :
        for angle in np.arange(6) :
    
            ax = field_sliced[:,:,field][angle]

            xx,yy = np.meshgrid(Y,X)
            zz = ax.reshape(len(Y), len(X))
            zz = np.where(zz < thresh * np.max(zz), 0, zz)
            zz = gaussian_filter(zz, sigma=2)
            
            axis[field,angle].imshow(zz,cmap='jet', origin='lower')
            if goal is not None :
                
                x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                
                axis[field, angle].scatter([x],[y],c='white',s=100)
            axis[field,angle].axis('off')
            fig.savefig(folder+'/mosaic_%s.png'%(m+1))

def scale_values(x,x_0,x_1,y_0,y_1) :
    return y_0 + (y_1 - y_0) * (x - x_0) / (x_1 - x_0)
            
def draw_ovc(fields, x_limits, y_limits,
                      grid_size, obj_x, obj_y, folder) :
    fields = np.load(fields)
    fields_avg = np.mean(fields, axis=0)
    
    n_fields = np.size(fields_avg,-1)
    n = np.int32(np.ceil(n_fields/5))
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])
    obj_x = scale_values(obj_x, x_limits[0], x_limits[1], 0, grid_size[0])
    obj_y = scale_values(obj_y, y_limits[0], y_limits[1], 0, grid_size[1])
    fig, axis = plt.subplots(n, 5, figsize=(12,12))
    axis = axis.ravel()
    
    for i in range(n_fields) : 
    
        ax = fields_avg[:,i]
        ax = gaussian_filter(ax, sigma=3)
        xx,yy = np.meshgrid(Y,X)
        zz = ax.reshape(len(Y), len(X))
        
        axis[i].imshow(zz,cmap='jet',origin='lower')
        axis[i].scatter([obj_x],[obj_y],c='green',s=50)
        axis[i].axis('off')

    fig.savefig(folder)
    
def plotReward(episodeReward, integrator=10) : 
    """ Returns the episode reward suitable for plotting by integrating over 
        the last n values as specified on the plot
        
    """
    y = []
    for i in range(len(episodeReward) - integrator):
        temp = np.sum(episodeReward[i:i+integrator])/integrator
        y.append(temp)
        
    x = np.arange(len(y))
    return np.array([x,y])


def mean_std_reward(training_curves) : 
    """ Returns mean and standard deviation of reward for multiple training
    runs """
    all_rewards = np.stack(([training_curves[i]['episode_reward'][:9999] for i in range(len(training_curves))]))
    mean_reward = np.mean(all_rewards, axis=0)
    std_dev = np.std(all_rewards, axis=0)
    
    return np.array([mean_reward,std_dev])

def create_metadata_txt(filename)  :
    file = open(filename,'w+')

    file.writelines(["ACT_SPACE : \n", 
                     "EXP_LENGTH : \n"
                     "INPUT_SIZE : \n"
                     "FIELD_VIEW : \n"
                     "SCENE_FILE : \n"
                     "ARENA_DIM : \n"
                     "GRAPH_SIZE : \n"
                     "GRAPH_TYPE : \n"
                     "GOAL_NODE : \n"
                     "START_NODE : \n"
                     "MEM_SIZE : \n"
                     "EPSILON : \n"
                     "LR : \n"
                     "NETWORK : \n"
                     "OTHER : "])
    
    file.close()
    
def write_metadata_json(filename_txt, filename_json) :
    
    data_dict ={}
    
    with open(filename_txt) as file :
        
        for line in file : 
            key, value = line.strip().split(':', 1)
            key = key.strip()
            data_dict[key] = value.strip()
            
    json_file = open(filename_json, "w")
    json.dump(data_dict, json_file, indent = 4, sort_keys=False)
    json_file.close()
    







