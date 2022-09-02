#templates for analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
import json
import itertools

from modules.base.WORLD_Modules import WORLD_BlenderInterface, WORLD_BlenderInterfaceWithRotation
from modules.base.TOP_Modules import TOP_HexGridModule, TOP_LinearTrack, TOP_HexGridOdorModule
from modules.base.OBS_Modules import OBS_ImageWorkingImage

from aux.helpers import generate_from_grid, resize_images, build_model
from aux.helpers import get_activity_map, limit_field_of_view, get_activity_map_odor, get_activity_map_image_mem
from aux.helpers import get_activity_map_lesion
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from keras.optimizers import Adam
from keras.models import model_from_json

save_location = 'data/fields/fields_2.npy'
#angles at which to calculate firing fields
angles = np.linspace(0.0,300.0,6,dtype='int64')

#---------------------------ALLOCENTRIC PLACE CELLS-------------------------#
def firing_fields_allo(metadata, weights, images, gen_input, angle, actions=8,
                       linear_track=False, memory_marker=False) :

    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    grid_size  = tuple(md['GRAPH_SIZE'])
    arena_size = tuple(md['ARENA_DIM'])
    goal_node  = md['GOAL_NODE']
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    x_limits = (-arena_size[0],arena_size[0])
    y_limits = (-arena_size[1],arena_size[1])
    if linear_track :
        y_limits = (0,0)
        resolution = (25,1)
    else : resolution = (25,25)
    #TODO : move this out or turn into parameter
    if gen_input :
        # load blender scene for input images 
        # and topology map to draw the goal node
        world = WORLD_BlenderInterface(scene)
        if linear_track :
            graph = TOP_LinearTrack(world,None,
           {'startNodes':[0],'goalNodes':[10], 'cliqueSize' :2, 'nodeCount':12,
            'start': (-2.0,0),'end':(2.0,0)},
            False)
        
        else :
            
            graph = TOP_HexGridModule(world,None,
               {'startNodes':[md['START_NODE']],'goalNodes':[None], 
                'cliqueSize' : 6, 'nodesXDomain':grid_size[0],'nodesYDomain':grid_size[1],
                'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]},
                False)

            
        # generates input images from a grid defined by resolution
        generate_from_grid(world, x_limits, y_limits, angle, resolution, 
                           images)
    x,y = None, None
    if goal_node is not None :
        x,y = graph.nodes[goal_node].x, graph.nodes[goal_node].y

    # load and process input images
    input_images   = np.load(images)  
    resized_images = resize_images(input_images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)

    # load json network from metadata
    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)

    # load and process RL agent
    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)

    # compute activity at each grid point
    if linear_track :
        activity = get_activity_map(cropped_images, model, dense_layer_idx=8,
                                    working_mem=True,mem_state=[[[2,0]]])
    #TODO :CHANGE!!!!
    #else : activity = get_activity_map(cropped_images, model)
    else : activity = get_activity_map_lesion(cropped_images, model, n_units=50)
    #activity = get_activity_map(cropped_images, model)
    activity = np.vstack(activity)[:,0,:]
    return activity, x, y

def firing_fields_allo_image(metadata, weights, images, gen_input, angle, actions=8,
                       linear_track=False, image_memory_from_node=0) :

    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    grid_size  = tuple(md['GRAPH_SIZE'])
    #arena_size = (6.0,6.0)
    arena_size = tuple(md['ARENA_DIM'])
    goal_node  = md['GOAL_NODE']
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    x_limits = (-arena_size[0],arena_size[0])
    y_limits = (-arena_size[1],arena_size[1])
    if linear_track :
        y_limits = (0,0)
        resolution = (25,1)
    else : resolution = (25,25)
    #TODO : move this out or turn into parameter
    if gen_input :
        # load blender scene for input images 
        # and topology map to draw the goal node
        world = WORLD_BlenderInterface(scene)
        if linear_track :
            graph = TOP_LinearTrack(world,None,
           {'startNodes':[0],'goalNodes':[10], 'cliqueSize' :2, 'nodeCount':12,
            'start': (-2.0,0),'end':(2.0,0)},
            False)
            image_obs = OBS_ImageWorkingImage(world, graph, None, False,
                                              input_size, fov)
            image_obs.createReferenceImages()
            mem_state = image_obs.observationFromNodeIndex(image_memory_from_node)

        else :

            graph = TOP_HexGridModule(world,None,
               {'startNodes':[md['START_NODE']],'goalNodes':[None], 
                'cliqueSize' : 6, 'nodesXDomain':grid_size[0],'nodesYDomain':grid_size[1],
                'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]},
                False)

            
        # generates input images from a grid defined by resolution
        generate_from_grid(world, x_limits, y_limits, angle, resolution, 
                           images)
    x,y = None, None
    if goal_node is not None :
        x,y = graph.nodes[goal_node].x, graph.nodes[goal_node].y

    # load and process input images
    input_images   = np.load(images)  
    resized_images = resize_images(input_images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)

    # load json network from metadata
    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)

    # load and process RL agent
    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)

    # compute activity at each grid point
    if linear_track :
        activity = get_activity_map_image_mem(cropped_images, model, mem_state=mem_state, dense_layer_idx=14)
    else : activity = get_activity_map(cropped_images, model)
    #activity = get_activity_map(cropped_images, model)
    activity = np.vstack(activity)[:,0,:]
    return activity, x, y

def firing_fields_marker_allo(metadata, weights, images, gen_input, angle, actions=8,
                              marker_id=None) :

    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    grid_size  = tuple(md['GRAPH_SIZE'])
    arena_size = tuple(md['ARENA_DIM'])
    start_nodes = [md['START_NODE']]
    goal_nodes  = [md['GOAL_NODE']]
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    y_limits = (0,0)
    #y_limits  = (-arena_size[1],arena_size[1])
    x_limits  = (-arena_size[0],arena_size[0])

    resolution = (25,1)

    if gen_input :
        # load blender scene for input images 
        # and topology map to draw the goal node
        world = WORLD_BlenderInterfaceWithRotation(scene)
        graph = TOP_LinearTrack(world,None,
           {'startNodes':[0],'goalNodes':[10], 'cliqueSize' :2, 'nodeCount':12,
            'start': (-2,0),'end':(2,0)},
            False)
# =============================================================================
#         graph    = TOP_HexGridModule(world,None,
#            {'startNodes':start_nodes,'goalNodes':goal_nodes, 'cliqueSize' :6, 'nodesXDomain':grid_size[0],
#             'nodesYDomain':grid_size[1],
#             'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]},
#             False)
# =============================================================================
        # generates input images from a grid defined by resolution
        x,y = None, None
        if marker_id is not None :
            x,y = graph.nodes[marker_id].x, graph.nodes[marker_id].y
            world.teleportXY('Cylinder',x,y)

        generate_from_grid(world, x_limits, y_limits, angle, resolution, images)

    # load and process input images
    input_images   = np.load(images)  
    resized_images = resize_images(input_images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)

    # load json network from metadata
    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)

    # load and process RL agent
    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)

    # compute activity at each grid point

    activity = get_activity_map(cropped_images, model)
    activity = np.vstack(activity)[:,0,:]
    return activity, x, y


def firing_fields_odor(metadata, weights, images, gen_input, angle, actions=12) :

    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    arena_size = tuple(md['ARENA_DIM'])
    goal_node  = md['GOAL_NODE']
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    x_limits = (-arena_size[0],arena_size[0])
    y_limits = (-arena_size[1],arena_size[1])
    resolution = (25,25)
    #TODO : move this out or turn into parameter
    if gen_input :
        # load blender scene for input images 
        # and topology map to draw the goal node
        world = WORLD_BlenderInterface(scene)
        graph = TOP_HexGridOdorModule(world,None,
           {'startNodes':[md['START_NODE']],'goalNodes':[None], 
            'cliqueSize' : 6, 'nodesXDomain':resolution[0],'nodesYDomain':resolution[1]+1,
            'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]},
            False)

            
        # generates input images from a grid defined by resolution
        generate_from_grid(world, x_limits, y_limits, angle, resolution, 
                           images)
    x,y = None, None
    if goal_node is not None :
        x,y = graph.nodes[goal_node].x, graph.nodes[goal_node].y

    # load and process input images
    input_images   = np.load(images)  
    resized_images = resize_images(input_images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)
    
    #odor signal
    odor_signal = []
    odor_source = graph.find_node(-2,0)
    graph.set_odor_source(odor_source)
    for node in graph.nodes :
        odor = graph.get_odor_signal(node.index)
        odor_signal.append(odor)
    odor_signal = odor_signal[:25]

    # load json network from metadata
    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)

    # load and process RL agent
    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)

    # compute activity at each grid point

    activity = get_activity_map_odor(cropped_images, model,odor_signal=None)
    activity = np.vstack(activity)[:,0,:]
    return activity, x, y

#-----------------------ALLOCENTRIC OBJECT VECTOR CELLS---------------------#
def firing_fields_ovc_allo(metadata, weights, images, angle, n_objects=3, 
                           obj_id='Cylinder', actions=8) :
    
    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    arena_size = tuple(md['ARENA_DIM'])
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    x_limits = (-arena_size[0],arena_size[0])
    y_limits = (-arena_size[1],arena_size[1])

    resolution = (25,25)

    world = WORLD_BlenderInterfaceWithRotation(scene)
   
    x_points = np.linspace(x_limits[0], x_limits[1], n_objects + 2)
    y_points = np.linspace(x_limits[0], x_limits[1], n_objects + 2)
    
    obj_positions = []
   
    for x, y in itertools.product(x_points,y_points) :
        if x > x_limits[0] and x < x_limits[1] and y > y_limits[0] and y < y_limits[1]  :
            obj_positions.append((y,x))

    # load json network from metadata
    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)
    
    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)


    fields = []
    for o in obj_positions : 
        world.teleportXY(obj_id, o[0], o[1])
        generate_from_grid(world, x_limits, y_limits, angle, resolution, 
                   images)
        input_images = np.load(images)  
        resized_images = resize_images(input_images, input_size[0], input_size[1])
        cropped_images = limit_field_of_view(resized_images, input_size[0], fov)
        activity = get_activity_map(cropped_images, model, dense_layer_idx=9)
        activity = np.vstack(activity)[:,0,:]
        fields.append(activity)

    return fields

#------------------------------AGENT CENTERED OVC---------------------------#
def firing_fields_ovc_agent(metadata, weights, images, angle, agent_pos=(0.0,0.0),
                            obj_id='Cylinder', actions=8) :
    
    with open(metadata) as fp :
        md = json.load(fp)
    
    # load variables from metadata file
    input_size = tuple(md['INPUT_SIZE'])
    arena_size = tuple(md['ARENA_DIM'])
    scene      = md['SCENE_FILE']
    network    = md['NETWORK']
    fov        = md['FIELD_VIEW']

    x_limits = (-arena_size[0],arena_size[0])
    y_limits = (-arena_size[1],arena_size[1])

    resolution = (25,25)
    world = WORLD_BlenderInterfaceWithRotation(scene)

    loaded_model_json = json.dumps(network)
    model = model_from_json(loaded_model_json)

    memory=SequentialMemory(limit=3000, window_length=1)
    policyEpsGreedy=EpsGreedyQPolicy(0.25)

    memory          = SequentialMemory(limit=md['MEM_SIZE'], window_length=1)
    policyEpsGreedy = EpsGreedyQPolicy(md['EPSILON'])
    agent           = DQNAgent(model=model, nb_actions=actions, memory=memory,
                 nb_steps_warmup=100, enable_dueling_network=True,
                 dueling_type='avg',
                 target_model_update=1e-2, policy=policyEpsGreedy)
    agent.compile(Adam(lr=md['LR']), metrics=['mae'])
    agent.load_weights(weights)

    fields = []

    X = np.linspace(x_limits[0], x_limits[1], resolution[0])
    Y = np.linspace(y_limits[0], y_limits[1], resolution[1])
    
    xx,yy = np.meshgrid(X,Y)
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T
        
    images = []
    
    
    for point in grid_points : 
        world.teleportXY(obj_id, point[0], point[1])
        observation = world.stepSimNoPhysics(agent_pos[0],agent_pos[1], angle)
        images.append(observation[3])
        
    resized_images = resize_images(images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)
    #TODO :CHANGE!
    #activity = get_activity_map(cropped_images, model, dense_layer_idx=9)
    activity = get_activity_map_lesion(cropped_images, model, n_units=50)
    activity = np.vstack(activity)[:,0,:]
    fields.append(activity)

    return fields

#------------------------------VISUALIZATION TOOLS--------------------------#
def draw_cell_mosaics(fields, fields_per_mosaic, x_limits, y_limits,
                      grid_size,folder, goal=None, thresh=0.65, rotated=False,
                      hds=6) :

    n_fields = np.size(fields,-1)
    n = divmod(n_fields,fields_per_mosaic)
    
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])
    
    start = 0
    for m in range(n[0]) :
        field_sliced = fields[:,:,start:start + fields_per_mosaic]
        fig, axis = plt.subplots(fields_per_mosaic,hds,figsize=(12,12))

        for field in np.arange(fields_per_mosaic) :
            #remove :
            #field_sliced[:,:, field] = np.where(field_sliced[:,:, field] < thresh * np.max(field_sliced[:,:, field]), 
            #            0, field_sliced[:,:, field])
            for angle in np.arange(hds) :

                ax = field_sliced[:,:,field][angle]
                
                xx,yy = np.meshgrid(Y,X)
                zz = ax.reshape(len(Y), len(X))
                #remove : 
                #zz = np.repeat(zz,3,0)
                zz = np.where(zz < thresh * np.max(zz), 0, zz)
                zz = gaussian_filter(zz, sigma=2)
                if rotated :
                    angles = [90, 30, 330, 270, 210, 150]
                    zz = rotate(zz,90.0 - angles[angle],reshape=False, cval = np.min(zz))
                axis[field,angle].imshow(zz,cmap='jet', origin='lower',vmin=0.0)
                if goal is not None :
                    x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                    y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                    
                    axis[field, angle].scatter([x],[y],c='white',s=100)
                axis[field,angle].axis('off')
                fig.savefig(folder+'/mosaic_%s.png'%m)
        start += fields_per_mosaic

    field_sliced = fields[:,:,start:start + n[1]]
    fig, axis = plt.subplots(n[1],hds,figsize=(12,12))

    for field in np.arange(n[1]) :
        #remove
        #field_sliced[:,:, field] = np.where(field_sliced[:,:, field] < thresh * np.max(field_sliced[:,:, field]), 
        #   0, field_sliced[:,:, field])
        for angle in np.arange(hds) :

            ax = field_sliced[:,:,field][angle]

            xx,yy = np.meshgrid(Y,X)
            zz = ax.reshape(len(Y), len(X))
            #remove : 
            #zz = np.repeat(zz,3,0)
            zz = np.where(zz < thresh * np.max(zz), 0, zz)
            zz = gaussian_filter(zz, sigma=2)
            
            axis[field,angle].imshow(zz,cmap='jet', origin='lower',vmin=0.0)
            if goal is not None :
                
                x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                
                axis[field, angle].scatter([x],[y],c='white',s=100)
            axis[field,angle].axis('off')
            fig.savefig(folder+'/mosaic_%s.png'%(m+1))
            
def draw_cell_mosaics_nempty(fields, fields_per_mosaic, x_limits, y_limits,
                      grid_size,folder, goal=None, thresh=0.65, rotated=False,
                      hds=2) :

    n_fields = np.size(fields,-1)
    n = divmod(n_fields,fields_per_mosaic)
    
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])
    
    start = 0
    for m in range(n[0]) :
        field_sliced = fields[:,:,start:start + fields_per_mosaic]
        fig, axis = plt.subplots(fields_per_mosaic,hds,figsize=(12,12))

        for field in np.arange(fields_per_mosaic) :
            for angle in np.arange(hds) :

                ax = field_sliced[:,:,field][angle]
                
                xx,yy = np.meshgrid(Y,X)
                zz = ax.reshape(len(Y), len(X))
                #remove : 
                zz = np.repeat(zz,3,0)
                zz = np.where(zz < thresh * np.max(zz), 0, zz)
                zz = gaussian_filter(zz, sigma=2)
                if rotated :
                    angles = [90, 30, 330, 270, 210, 150]
                    zz = rotate(zz,90.0 - angles[angle],reshape=False, cval = np.min(zz))
                axis[field,angle].imshow(zz,cmap='jet', origin='lower')
                if goal is not None :
                    x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                    y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                    
                    axis[field, angle].scatter([x],[y],c='white',s=100)
                axis[field,angle].axis('off')
                fig.savefig(folder+'/mosaic_%s.png'%m)
        start += fields_per_mosaic

    field_sliced = fields[:,:,start:start + n[1]]
    fig, axis = plt.subplots(n[1],hds,figsize=(12,12))

    for field in np.arange(n[1]) :
        for angle in np.arange(hds) :

            ax = field_sliced[:,:,field][angle]

            xx,yy = np.meshgrid(Y,X)
            zz = ax.reshape(len(Y), len(X))
            #remove : 
            zz = np.repeat(zz,3,0)
            zz = np.where(zz < thresh * np.max(zz), 0, zz)
            zz = gaussian_filter(zz, sigma=2)
            
            axis[field,angle].imshow(zz,cmap='jet', origin='lower')
            if goal is not None :
                
                x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                
                axis[field, angle].scatter([x],[y],c='white',s=100)
            axis[field,angle].axis('off')
            fig.savefig(folder+'/mosaic_%s.png'%(m+1))

def draw_cell_mosaics_linear(fields, fields_per_mosaic, x_limits, y_limits,
                      grid_size,folder, goal=None, thresh=0.65, rotated=False) :

    n_fields = np.size(fields,-1)
    n = divmod(n_fields,fields_per_mosaic)
    
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])
    
    start = 0
    for m in range(n[0]) :
        field_sliced = fields[:,:,start:start + fields_per_mosaic]
        fig, axis = plt.subplots(fields_per_mosaic,1,figsize=(1,12))

        for field in np.arange(fields_per_mosaic) :

            ax = field_sliced[:,:,field]
            
            xx,yy = np.meshgrid(Y,X)
            zz = ax.reshape(len(Y), len(X))
            zz = np.where(zz < thresh * np.max(zz), 0, zz)
            zz = gaussian_filter(zz, sigma=2)

            axis[field].imshow(zz,cmap='jet', origin='lower')
            if goal is not None :
                x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
                y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
                
                axis[field].scatter([x],[y],c='white',s=100)
            axis[field].axis('off')
            fig.savefig(folder+'/mosaic_%s.png'%m)
        start += fields_per_mosaic

    field_sliced = fields[:,:,start:start + n[1]]
    fig, axis = plt.subplots(n[1],1,figsize=(12,12))

    for field in np.arange(n[1]) :


        ax = field_sliced[:,:,field]

        xx,yy = np.meshgrid(Y,X)
        zz = ax.reshape(len(Y), len(X))
        zz = np.where(zz < thresh * np.max(zz), 0, zz)
        zz = gaussian_filter(zz, sigma=2)
        
        axis[field].imshow(zz,cmap='jet', origin='lower')
        if goal is not None :
            
            x = scale_values(goal[0], x_limits[0], x_limits[1], 0, grid_size[0])
            y = scale_values(goal[1], y_limits[0], y_limits[1], 0, grid_size[1])
            
            axis[field].scatter([x],[y],c='white',s=100)
        axis[field].axis('off')
        fig.savefig(folder+'/mosaic_%s.png'%(m+1))
            
def draw_ovc_allocentric(fields, fields_per_mosaic, x_limits, y_limits,
                      grid_size,folder, n_objects=3, thresh=0.5, hd=None):

    if hd is not None :
        idx = int(hd/60)
        fields  = fields[idx]
    else :
        fields  = np.mean(fields, axis=0)

    columns = len(fields)
    rows    = np.size(fields, axis=2)
    n_plots = divmod(rows, fields_per_mosaic)

    X = np.linspace(x_limits[0], x_limits[1], resolution[0])
    Y = np.linspace(y_limits[0], y_limits[1], resolution[1])

    x_points = np.linspace(x_limits[0], x_limits[1], n_objects + 2)
    y_points = np.linspace(x_limits[0], x_limits[1], n_objects + 2)
    
    obj_positions = []
       
    for x, y in itertools.product(x_points,y_points) :
        if x > x_limits[0] and x < x_limits[1] and y > y_limits[0] and y < y_limits[1]  :
            obj_positions.append((x,y))

    start_cell = 0
    for plot in range(n_plots[0]):
        fields_s = fields[:,:,start_cell:start_cell + fields_per_mosaic]
        fig, axis = plt.subplots(fields_per_mosaic, columns, figsize=(15,fields_per_mosaic*1.625))
        fig.suptitle('Head direction = %s'%hd, fontsize=16)
        fig.text(0.5, 0.04, 'object positions', ha='center', va='center', fontsize=16)
        fig.text(0.06, 0.5, 'cells', ha='center', va='center', rotation='vertical', fontsize=16)
        for r in np.arange(fields_per_mosaic) :
            for c in np.arange(columns) :
                ax = fields_s[:,:,r][c]
                
                xx,yy = np.meshgrid(Y,X)
                zz = ax.reshape(len(Y), len(X))
                zz = np.where(zz < 0.50 * np.max(zz), 0, zz)
                zz = gaussian_filter(zz, sigma=2)       
                axis[r,c].imshow(zz,cmap='jet', origin='lower')
                
                x = scale_values(obj_positions[c][0], x_limits[0], x_limits[1], 0, resolution[0])
                y = scale_values(obj_positions[c][1], y_limits[0], y_limits[1], 0, resolution[1])
                axis[r, c].scatter([x],[y],c='white',s=50)
                axis[r,c].axis('off')
                fig.savefig(folder+'/mosaic_%s.png'%plot)

        start_cell += fields_per_mosaic

    fields_s = fields[:,:,start_cell:start_cell + n_plots[1]]
    fig, axis = plt.subplots(n_plots[1], columns, figsize=(12,n_plots[1]*1.625))
    fig.suptitle('Head direction = %s'%hd, fontsize=16)
    fig.text(0.5, 0.04, 'object positions', ha='center', va='center', fontsize=16)
    fig.text(0.06, 0.5, 'cells', ha='center', va='center', rotation='vertical', fontsize=16)
    
    for r in np.arange(n_plots[1]) :
        for c in np.arange(columns) :
            ax = fields_s[:,:,r][c]
    
            xx,yy = np.meshgrid(Y,X)
            zz = ax.reshape(len(Y), len(X))
            zz = np.where(zz < 0.50 * np.max(zz), 0, zz)
            zz = gaussian_filter(zz, sigma=2)         
            axis[r,c].imshow(zz,cmap='jet', origin='lower')
            x = scale_values(obj_positions[c][0], x_limits[0], x_limits[1], 0, grid_size[0])
            y = scale_values(obj_positions[c][1], y_limits[0], y_limits[1], 0, grid_size[1])
            axis[r, c].scatter([x],[y],c='white',s=50)
            axis[r,c].axis('off')
            fig.savefig(folder+'/mosaic_%s.png'%(plot+1))


def draw_ovc_polar() : 
    pass

def scale_values(x,x_0,x_1,y_0,y_1) :
    return y_0 + (y_1 - y_0) * (x - x_0) / (x_1 - x_0)


