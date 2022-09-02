import numpy as np
import os
from cobel.spatial_representations.topology_graphs.simple_topology_graph import HexagonalGraph
from cobel.frontends.frontends_blender import FrontendBlenderInterface
import pyqtgraph as qg
from tensorflow.keras import backend as K
from custom_modules.observations.vector_observations import VectorObservation
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from cobel.agents.dqn_agents import DQNAgentBaseline

move_goal_at = 10

def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    | **Args**
    | values: A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = -1.0
    end_trial = False
    
    if values['currentNode'].goalNode:
        reward = 10.0
        end_trial = True
    
    return reward, end_trial

def move_goal_callback(logs) :
    trial    = logs['trial']
    rl_agent = logs['rl_parent']
    topology = rl_agent.interface_OAI.modules['spatial_representation']
    if trial%move_goal_at == 0 : 
        node = topology.select_random_nodes(n_nodes=1)
        topology.set_goal_nodes(node)
    
def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    visual_output = True
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Vector Navigation')
        
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('emergent_spatial_representations')[0] + '/cobel/environments/environments_blender/simple_grid_graph_maze.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = VectorObservation(None, main_window,vector_encoding='egocentric')
    modules['spatial_representation'] = HexagonalGraph(n_nodes_x=10, n_nodes_y=10,
                                                       n_neighbors=6, goal_nodes=[11],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True)
    modules['observation'].add_topology_graph(modules['spatial_representation'])
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, reward_callback)
        
    # amount of trials
    number_of_trials = 200
    # maximum steos per trial
    max_steps = 30
    
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [-30, 10])
    
    # initialize RL agent
    rl_agent = DQNAgentBaseline(modules['rl_interface'], 1000000, 0.3, None, 
                                custom_callbacks={'on_trial_end': [reward_monitor.update, move_goal_callback]})
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stopBlender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':

    single_run()
    # clear keras session (for performance)
    K.clear_session()
