# basic imports
from time import strftime, gmtime
import os
import numpy as np
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
# framework imports

from custom_modules.renderers.renderers import BlenderOnlineRenderer
from custom_modules.observations.image_observations import ImageObservationFOV
from custom_modules.spatial_representations.hexagonal_topology import HexagonalGraphExtended
from cobel.agents.dqn_agents import DQNAgentBaseline
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True
move_goal_at = 10
network = "../../networks/full_model.json"
json_file   = open(network, 'r')
loaded_model_json = json_file.read()
json_file.close()  
model = model_from_json(loaded_model_json)    
print(model.summary())

def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    | **Args**
    | values: A dict of values that are transferred from the OAI module to the 
      reward function. This is flexible enough to accommodate for different 
      experimental setups.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = False
    
    if values['currentNode'].goalNode:
        reward = 1.0
        end_trial = True
    
    return reward, end_trial

def move_goal_callback(logs) :
    trial    = logs['trial']
    rl_agent = logs['rl_parent']
    topology = rl_agent.interface_OAI.modules['spatial_representation']
    world    = rl_agent.interface_OAI.modules['world']
    if trial%move_goal_at == 0 : 
        node = topology.select_random_nodes(n_nodes=1)
        x,y  = topology.nodes[node[0]].x, topology.nodes[node[0]].y
        world.teleportXY('Cylinder',x,y)
        topology.set_goal_nodes(node)

def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''

        
    ts = strftime('%Y%m%d_%H%M',gmtime())

    training_file = 'training_%s.npy'%ts
    weights_file  = 'weights_%s.h5f'%ts
    actions_file  = 'actions_%s.npy'%ts
    
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Aiming')
        
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('emergent_spatial_representations')[0] + '/emergent_spatial_representations/worlds/aiming.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = BlenderOnlineRenderer(demo_scene)
    modules['observation'] = ImageObservationFOV(modules['world'], main_window, visual_output,
                                                 imageDims=(72, 12), view_angle=240.0)
    modules['spatial_representation'] = HexagonalGraphExtended(n_nodes_x=9, n_nodes_y=9,
                                                       n_neighbors=6, goal_nodes=[38],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True)
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, reward_callback)
    
    # amount of trials
    number_of_trials = 5000
    # maximum steps per trial
    max_steps = 100
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, 
                                   [-max_steps, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 3000, 0.3, model=model, 
                                custom_callbacks={'on_trial_end': [reward_monitor.update,
                                                                   move_goal_callback]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    if not os.path.exists('data_aiming') :
        os.mkdir('data_aiming')
    data = rl_agent.history.history
    actions = modules['rl_interface'].get_actions()
    rl_agent.agent.save_weights('data_aiming/'+ weights_file)
    
    #save training data
    np.save('data_aiming/'+ training_file, data)
    np.save('data_aiming/'+ actions_file, actions)
    # clear keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stopBlender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    for _ in range(15) :    
        single_run()
        # clear keras session (for performance)
        K.clear_session()