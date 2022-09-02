import numpy as np
import gym

class NoiseInjection() : 
    
    def __init__(self, gui_parent, lesion_unit=None, lesion_mag=0.0,
                 activation_file=None) :
        self.gui_parent  = gui_parent
        self.lesion_unit = lesion_unit
        self.lesion_mag  = lesion_mag
        self.activation_file = activation_file
        self.avg_max_acts = None

    def update(self) : 
        '''
        Updates the observation. 

        Returns
        -------
        None.

        '''
        noise = np.zeros(50)
        
        if self.activation_file is not None : 
            self.avg_max_acts = np.load(self.activation_file)
            self.cutoff       = np.mean(self.avg_max_acts)
            if self.lesion_unit is not None :
                if self.lesion_unit.shape!=() : 
                    for l in self.lesion_unit : 
                        val = np.min([self.avg_max_acts[l],self.cutoff])

                        if val==0 : 
                            val = 1.0
                        noise[l] = np.abs(np.random.normal(scale = val * self.lesion_sigma))
                        print("injecting noise ... ", noise[l])
                else : 
                    val = np.min([self.avg_max_acts[self.lesion_unit],self.cutoff])
                    if val==0:
                        val=1.0
                    noise[self.lesion_unit] = np.abs(np.random.normal(scale = val * self.lesion_sigma))
                    print("injecting noise ... ", noise[self.lesion_unit])

        self.observation = noise
            
    def getObservationSpace(self):
        
        '''
        This function returns the observation space for the given observation class.
        '''
        return gym.spaces.Box (low=0.0, high=10.0, shape=(50,))