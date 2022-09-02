''' Generate input images on a grid for
    recording activations
'''
from aux.helpers import resize_images, limit_field_of_view
import numpy as np
import json
from modules.base.WORLD_Modules import WORLD_BlenderInterface_angle

metadata_file = "metadata_aiming.json"
with open(metadata_file) as fp :
    md = json.load(fp)

scene = md['SCENE_FILE']
arena_size = tuple(md['ARENA_DIM'])
input_size = tuple(md['INPUT_SIZE'])
fov        = md['FIELD_VIEW']
x_limits = (-arena_size[0],arena_size[0])
y_limits = (-arena_size[1],arena_size[1])

def generate_from_grid(world, x_limits, y_limits, angle, grid_size,
                       obj_id='Cylinder') :
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
        world.teleportXY(obj_id, point[0], point[1])
        observation = world.stepSimNoPhysics(0.0,0.0, angle)
        images.append(observation[3])
        
    resized_images = resize_images(images, input_size[0], input_size[1])
    cropped_images = limit_field_of_view(resized_images, input_size[0], fov)

    return cropped_images


angles = np.linspace(0,300,6,dtype='int64')
resolution = (25,25)
data = []
world = WORLD_BlenderInterface_angle(scene)
for a in angles :
    data.append(generate_from_grid(world, x_limits, y_limits, a, resolution))
    
np.save('input_images_aiming.npy',data)





