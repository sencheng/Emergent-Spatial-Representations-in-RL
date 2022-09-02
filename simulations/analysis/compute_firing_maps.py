from aux.analysis_templates import firing_fields_allo, firing_fields_ovc_agent
import numpy as np
import os

path = '../full_model/data_guidance/'
#path = '../full_model/data_aiming/'

files = os.listdir(path)

metadata = [f for f in files if f.startswith('metadata.json')]
weights  = [f for f in files if f.startswith('weights')]

weights.sort()

def compute_firing_maps(coord='xy') :
    
    for idx in range(len(weights)) :
        
        suffix = weights[idx].replace('weights','')    
        angles = np.linspace(0.0,300.0,6,dtype='int64') #head directions for which to compute maps
        
        fields = []
        for a in angles : 
            if coord == 'xy' :
                firing_fields, x, y = firing_fields_allo(metadata[0], weights[idx],
                                                         'input.npy', True, a, actions=12)
        
                fields.append(firing_fields)
            
            if coord == 'agent' : 
                firing_fields = firing_fields_ovc_agent(metadata[0], weights[idx],
                                                        'input.npy', a, actions=3)
                fields.append(firing_fields)
            
            if coord == 'radial' : 
                firing_fields = firing_fields_ovc_agent(metadata[0], weights[idx],
                                                        'input.npy', a, actions=3,
                                                        radial=True)
                fields.append(firing_fields)
        
        data = {'FIELDS' : fields, 'GOAL' : np.array([x,y])}
        np.save('fields%s'%suffix, data)
        os.remove('input.npy')

if __name__ == '__main__':  
    
    compute_firing_maps(coord='xy') #use coord='agent' for agent centered and 
                                    #'radial' for radial firing maps