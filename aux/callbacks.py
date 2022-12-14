import numpy as np
import pyqtgraph as qg

visualOutput = True
class performanceMonitor():
    '''
    Draw the performance curve (reinforcement learning reward) as the
    experiment progresses.
    '''
    def __init__(self, rlAgent, visualOutput):
        
        self.rlAgent = rlAgent        
        self.visualOutput = visualOutput
        #TODO
        self.rlRewardTraceRaw = np.zeros(rlAgent.exp_length,
                                         dtype='float')
        self.rlRewardTraceRefined = np.zeros(rlAgent.exp_length,
                                             dtype='float')
        self.accumulationRangeReward = 20
                
        if visualOutput:
            rlAgent.guiParent.setGeometry(0,0,1920,600)
            

            self.rlRewardPlot = rlAgent.guiParent.addPlot(title="Reinforcement learning progress")
            self.rlRewardPlot.setXRange(0,rlAgent.exp_length)
            self.rlRewardPlot.setYRange(-100.0,100.0)
            self.episodesDomain = np.linspace(0,rlAgent.exp_length,
                                            rlAgent.exp_length)
            self.rlRewardTraceRawGraph=self.rlRewardPlot.plot(self.episodesDomain,
                                                              self.rlRewardTraceRaw)
            self.rlRewardTraceRefinedGraph=self.rlRewardPlot.plot(self.episodesDomain,
                                                                  self.rlRewardTraceRefined)

                    
    def clearPlots(self):
        '''
        This function clears the plots generated by the performance monitor.
        '''
        if self.visualOutput:
            self.rlAgent.guiParent.removeItem(self.rlRewardPlot)
    
    
    def update(self,trial,logs):
        '''
        This function is called when a trial ends. Here, information about the 
        monitored variables is memorized, and the monitor graphs are updated.
        
        trial:  the actual trial number
        logs:   information from the reinforcement learning subsystem
        '''
        
        rlReward = logs['episode_reward']
        self.rlRewardTraceRaw[trial] = rlReward        
        aggregatedRewardTraceRaw = None
        
        if trial < self.accumulationRangeReward:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:None:-1]
        else:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:trial-self.accumulationRangeReward:-1]
        self.rlRewardTraceRefined[trial] = np.mean(aggregatedRewardTraceRaw)
        
        if self.visualOutput:
            self.rlRewardTraceRawGraph.setData(self.episodesDomain,
                                               self.rlRewardTraceRaw,
                                               pen=qg.mkPen(color=(128,128,128),
                                                            width=1))
            self.rlRewardTraceRefinedGraph.setData(self.episodesDomain,
                                                   self.rlRewardTraceRefined,
                                                   pen=qg.mkPen(color=(255,0,0),
                                                                width=2))
        
        
def rewardCallback(values):
    
    rlAgent=values['rlAgent']
                
    reward = -1
    stopEpisode = False

    nodes=values['modules']['topologyModule'].nodes
    
    if values['currentNode'].goalNode:
        reward = 1.0
        rlAgent.performanceValues['finalNode'] = values['currentNode'].index
        stopEpisode = True
        
    
    
    reward += nodes[values['currentNode'].index].nodeRewardBias
    
    return [reward,stopEpisode]


def trialBeginCallback(trial,rlAgent):
    '''
    This is a callback function that is called in the beginning of each trial. 
    Here, experimental behavior can be defined.
    
    trial:      the number of the finished trial
    rlAgent:    the employed reinforcement learning agent
    '''    


    if trial == rlAgent.exp_length - 1:
        rlAgent.agent.step=rlAgent.maxSteps+1
    
    
def trialEndCallback(trial,rlAgent,logs):
    '''
    This is a callback routine that is called when a single trial ends. Here, 
    functionality for performance evaluation can be introduced.
    trial:      the number of the finished trial
    rlAgent:    the employed reinforcement learning agent
    logs:       output of the reinforcement learning subsystem
    '''
    rlAgent.perfMon.update(trial,logs)
    
    if visualOutput:
        rlAgent.OAIInterface.modules['topologyModule'].updateVisualElements()
