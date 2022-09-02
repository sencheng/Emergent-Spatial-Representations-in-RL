import shutil 
import os
import numpy as np
import pyqtgraph as qg
import json
from keras import backend
from time import strftime, gmtime

from aux.callbacks import rewardCallback, trialEndCallback
from aux.callbacks import  performanceMonitor

from modules.base.OAI_Mixed_Modules import OAIGymInterfaceAllocentricLesions
from modules.base.WORLD_Modules import WORLD_BlenderInterface_angle
from modules.base.EXE_Modules import EXE_TeleportationPlanningModule
from modules.base.MC_Modules import MC_TeleportationActuatorModule

# =============================================================================
# from modules.base.OAI_Mixed_Modules import OAIGymInterfaceEgocentricLesions
# from modules.base.WORLD_Modules import WORLD_BlenderInterfaceWithRotation
# from modules.base.EXE_Modules import EXE_RotationPlanningModule
# from modules.base.MC_Modules import MC_RotationActuatorModule
# =============================================================================

from modules.base.TOP_Modules import TOP_HexGridModule
from modules.base.ACT_Mixed_Modules import ACT_ReinforcementLearningModule
from modules.base.OBS_Modules import OBS_ImageLesions
from modules.base.OBS_Modules import OBS_ImageLesionsFromFile
from modules.base.MAP_Modules import MAP_GroundTruthCognitiveMapModule

visualOutput = True
trial_length_goal = 5

def calc_other_ids(class_ids, n_units=50):
    ''' class ids is a list of the indices of units in the recurrent layer 
    that have been classified as different cell types. This is a list of 
    5 numpy arrays which are as follows:
        0 - place like
        1 - no activity
        2 - partially active
        3 - direction modulated
        4 - vector like
    This function returns the cell indices that are not classified as any of 
    these types as a numpy array
    '''
    all_ids    = np.arange(0, n_units)
    classified = np.concatenate([c if c.shape!=() else c.reshape(1,) for c in class_ids])
    other_ids  = np.setdiff1d(all_ids, classified)
    return other_ids

def get_data_from_exp(classified_fields, exp_idx) : 
    exp_data = []
    for c in classified_fields : 
        exp_data.append(c[exp_idx])
    return exp_data

def run_experiment(weights,metadata, lesioned_unit, lesion_mag, acts):
    np.random.seed()
    with open(metadata) as fp :
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
    mainWindow = None
    if visualOutput:
        mainWindow = qg.GraphicsWindow( title="Emergent Spatial Representations" )

#---------------------------CREATE MODULES----------------------------------#
    def trialBeginCallback(trial,rlAgent) :
        """redefinition of the callback function called at the beginning
        of each trial to accommodate aiming for an object """
        if trial%trial_length_goal == 0:
    
            idx, x, y = rlAgent.OAIInterface.modules['topologyModule'].select_random_node()
            
            rlAgent.OAIInterface.modules['worldModule'].teleportXY('Cylinder',x,y)
            rlAgent.OAIInterface.modules['topologyModule'].setGoal(idx)
            rlAgent.OAIInterface.modules['observationModule'].referenceImages=[]
            rlAgent.OAIInterface.modules['observationModule'].createReferenceImages(rlAgent.OAIInterface.modules['topologyModule'])
    
            
        if trial == exp_length:
            rlAgent.agent.step=rlAgent.maxSteps+1

    modules = dict()    
    modules['worldModule']       = WORLD_BlenderInterface_angle(scene)
    modules['observationModule'] = OBS_ImageLesionsFromFile(modules['worldModule'],mainWindow,
           visualOutput,image_dims = input_size,view_angle=fov,lesion_unit=lesioned_unit,
           lesion_sigma=lesion_mag,avg_max_act_file=acts)
    modules['mapModule']         = MAP_GroundTruthCognitiveMapModule(modules['worldModule'],mainWindow)    
    modules['topologyModule']    = TOP_HexGridModule(modules['worldModule'],mainWindow,
           {'startNodes':start_nodes,'goalNodes':goal_nodes, 'cliqueSize' :6, 'nodesXDomain':grid_size[0],
            'nodesYDomain':grid_size[1],
            'gridPerimeter':[-arena_size[0],arena_size[0],-arena_size[1],arena_size[1]]},
            visualOutput)
    modules['observationModule'].topologyModule = modules['topologyModule']
    modules['vectorPlanningModule'] = EXE_TeleportationPlanningModule(modules['worldModule'],mainWindow)
    modules['actuatorModule'] = MC_TeleportationActuatorModule()
    OAIInterface = OAIGymInterfaceAllocentricLesions(modules, visualOutput, rewardCallback)

#---------------------------SIGNAL SLOT CONNECTION--------------------------#

    modules['worldModule'].sig_robotPoseChanged.connect(modules['mapModule'].receiveRobotPoseData) 
    modules['mapModule'].sig_MAP_poseEstimateAvailable.connect(modules['topologyModule'].updateRobotPose) 
    modules['mapModule'].sig_MAP_poseEstimateAvailable.connect(modules['observationModule'].updateRobotPose) 
    modules['vectorPlanningModule'].sig_motorCommandProposal.connect(modules['actuatorModule'].assembleActuatorCommand)
    modules['actuatorModule'].sig_finalActuatorCommand.connect(modules['worldModule'].actuateRobot)
    modules['worldModule'].sig_triggerVectorBasedPlanning.connect(modules['vectorPlanningModule'].setGoalPosition)
    modules['observationModule'].sig_observationAvailable.connect(OAIInterface.updateObservation) 


    rlAgent = ACT_ReinforcementLearningModule(mainWindow,OAIInterface,{'agentType':'DQN'},
                                               visualOutput,exp_length,
                                               trialBeginCallback, replay_mem, eps,trialEndCallback,network)
    rlAgent.perfMon = performanceMonitor(rlAgent,visualOutput)
    rlAgent.infoValues = dict()
    rlAgent.infoValues['possibleStartNodes'] = modules['topologyModule'].graphInfo['startNodes'].copy()
    np.random.shuffle(rlAgent.infoValues['possibleStartNodes'])

    OAIInterface.rlAgent = rlAgent
    modules['topologyModule'].rlAgent = rlAgent

    data = rlAgent.test(50,25,weights,save=None)
    rlAgent.perfMon.clearPlots()


#------------------------------CLEAR PROCESSES------------------------------#
    backend.clear_session()
    modules['worldModule'].stopBlender()
    
    return data

if __name__ == "__main__":
    
    s = ['_20210131_1408','_20210131_1420','_20210131_1434','_20210131_1448','_20210131_1543']
    for suffix in s : 
        classified_fields = np.load('new_field_classification'+suffix+'.npy')
        classified_fields = classified_fields.tolist()
        class_ids = get_data_from_exp(classified_fields,0)
        other_ids = calc_other_ids(class_ids)
        lesion_mags = np.arange(0,5,1)
        for l in lesion_mags : 
            for i in range(5) : 
                ids = class_ids[i]
                test_data = []
                if ids.size == 0 :
                    ids = None
                print(ids)
                data = run_experiment('weights'+suffix+'.h5f',
                                      'metadata'+suffix+'.json',
                                      lesioned_unit=ids,
                                      lesion_mag=l,acts = 'avg'+suffix+'.npy')
                test_data.append(data)
                np.save('test'+suffix+'/test_pop_'+str(i)+'_'+str(l)+'.npy', test_data)
            test_data = []
            data = run_experiment('weights'+suffix+'.h5f',
                                  'metadata'+suffix+'.json',
                                  lesioned_unit=other_ids,
                                  lesion_mag=l,acts = 'avg'+suffix+'.npy')
            test_data.append(data)
            np.save('test'+suffix+'/test_pop_'+str(5)+'_'+str(l)+'.npy', test_data)
