import numpy as np
import os

files = os.listdir()
field_files = [f for f in files if f.startswith('fields_')]
field_files.sort()

for f in field_files : 
    fields    = np.array(np.load(f).item()['FIELDS'])
    save_name = f.replace('fields_', 'avg_activation_')
    max_values = np.zeros((50,6))
    avg_activation = np.zeros(50)
    for i in range(50) : 
        field = fields[:,:,i]
        avg_activation[i] = np.mean(field)
        for j in range(6) : 
            max_values[i][j] = np.max(field[j])
            
    average_max_activation = np.mean(max_values, axis=1)
    mean_grand = np.mean(avg_activation)
    np.save(save_name,avg_activation)
