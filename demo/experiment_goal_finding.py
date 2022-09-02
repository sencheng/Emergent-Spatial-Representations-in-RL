
# basic imports
import shutil
import multiprocessing as mp
import numpy as np
import json
from time import strftime, gmtime

import pyqtgraph as qg

from multiprocessing import Pool, Process
from pathlib import Path
from numpy import random
from keras import backend

from frontends.frontends_blender import FrontendBlenderInterface
from spatial_representations.topology_graphs.hex_no_rotation import HexTopologyAllocentric
from agents.dqn_agents import DQNAgentBaseline
from observations.image_observations import ImageObservationFOV
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline


# shall the system provide visual output while performing the experiments? NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visualOutput=True



'''
This is a callback function that defines the reward provided to the robotic agent. Note: this function has to be adopted to the current experimental design.

values: a dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
'''
def rewardCallback(values):
    # the standard reward for each step taken is negative, making the agent seek short routes
    
    rlAgent=values['rlAgent']
        
    reward=-1.0
    stopEpisode=False
    
    
    
    if values['currentNode'].goalNode:
        reward=10.0
        stopEpisode=True
    
    
    return [reward,stopEpisode]




'''
This is a callback function that is called in the beginning of each trial. Here, experimental behavior can be defined (ABA renewal and the like).

trial:      the number of the finished trial
rlAgent:    the employed reinforcement learning agent
'''
def trialBeginCallback(trial,rlAgent):
        
    if trial==rlAgent.trialNumber-1:
        
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step=rlAgent.maxSteps+1
    
    


'''
This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation can be introduced.
trial:      the number of the finished trial
rlAgent:    the employed reinforcement learning agent
logs:       output of the reinforcement learning subsystem
'''

def trialEndCallback(trial,rlAgent,logs):

    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements()
        rlAgent.performanceMonitor.update(trial,logs)

'''
This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output), or by a direct call (in this case, visual output can be used).

combinations:           this is a combination of parameters used for a single experiment. Note: the combination values depend on the experimental design!
'''

def singleRun(metadata):
    
    np.random.seed()
    with open(metadata) as fp:
        md = json.load(fp)
        
    #---------------------------METADATA FOR EXPERIMENT-------------------------#
    scene      = md['SCENE_FILE']
    replay_mem  = md['MEM_SIZE']
    eps         = md['EPSILON']
    network     = md['NETWORK']
    exp_length  = md['EXP_LENGTH']
    start_nodes = [md['START_NODE']]
    goal_nodes  = [md['GOAL_NODE']]
    input_size  = tuple(md['INPUT_SIZE'])
    fov         = md['FIELD_VIEW']
    grid_size   = tuple(md['GRAPH_SIZE'])
    arena_size  = tuple(md['ARENA_DIM'])

#---------------------------SAVE LOCATIONS----------------------------------#
    ts = strftime('%Y%m%d_%H%M',gmtime())

    training_file = 'training_%s.npy'%ts
    weights_file  = 'weights_%s.h5f'%ts

#---------------------------GRAPHICS WINDOW---------------------------------#
    
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    mainWindow=None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = qg.GraphicsWindow( title="Spatial Representations" )
    

    # a dictionary that contains all employed modules
    modules=dict()

    modules['world']=FrontendBlenderInterface(scene)
    modules['observation']=ImageObservationFOV(modules['world'],mainWindow,visualOutput,
           image_dims=input_size, view_angle=fov)
    modules['spatial_representation']=HexTopologyAllocentric(modules,{'startNodes':start_nodes,
           'goalNodes':goal_nodes,'cliqueSize':6, 'nodesXDomain':grid_size[0],
           'nodesYDomain':grid_size[1], 
           'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]})
    modules['spatial_representation'].set_visual_debugging(visualOutput,mainWindow)
    modules['rl_interface']=OAIGymInterface(modules,visualOutput,rewardCallback)

    #rlAgent=DQNAgentLoadNetwork(modules['rl_interface'],replay_mem,eps,trialBeginCallback,trialEndCallback,0.001,network)
    rlAgent=DQNAgentBaseline(modules['rl_interface'],memoryCapacity=replay_mem,
                                epsilon=eps,processor=None,trialBeginFcn=trialBeginCallback,
                                trialEndFcn=trialEndCallback,lr=0.001,network=network)
    # set the experimental parameters
    rlAgent.trialNumber=100 #set too exp_length
    
    perfMon=RLPerformanceMonitorBaseline(rlAgent,mainWindow,visualOutput)
    rlAgent.performanceMonitor=perfMon
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent=rlAgent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent=rlAgent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rlAgent.train(100000)
    
    backend.clear_session()
    modules['world'].stopBlender()


if __name__ == "__main__":
    singleRun('metadata.json')

