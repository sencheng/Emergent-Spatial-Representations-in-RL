import numpy as np
import os
from field_classification_helpers import place_like, vector_cells, empty_fields
from field_classification_helpers import modulated

def classify_modulated_vector(idx_modulated, idx_vector) :
    mask         = np.in1d(idx_modulated,idx_vector)
    mask2        = np.in1d(idx_vector,idx_modulated)
    id_modulated = idx_modulated[~mask]
    id_vector    = idx_modulated[mask]
    return mask, mask2, id_modulated, id_vector


def reclassify(data_modulated, data_vector) :
    for m,v in zip(data_modulated,data_vector) :
        mask, mask2, ids_mod, ids_vec = classify_modulated_vector(m[1], v[1])
        m[1] = ids_mod
        v[1] = ids_vec
        m[0] = (m[0])[~mask]
        v[0] = (v[0])[mask2]

n_units = 50


def avg_std_field_types(loc, trials, total_fields) : 
    subfolders = [ f.name for f in os.scandir(loc) if f.is_dir()]
    subfolders.sort()
    
    data_place     = []
    data_vector    = []
    data_empty     = []
    data_modulated = []
    
    for s in subfolders : 
        field = np.load(loc+s+'/'+str(trials)+'.npy')
        field = np.squeeze(field).reshape((6,625,n_units))
        
        fields,ids,pd,v = place_like(field)
        data_place.append([fields, ids, pd, v])
        
        fields,ids,pd,v = vector_cells(field)
        data_vector.append([fields,ids,pd,v])
        
        fields,ids,pd,v = modulated(field)
        data_modulated.append([fields,ids,pd,v])
        
        ids_all, ids_any = empty_fields(field)
        data_empty.append([ids_all, ids_any])
    
    reclassify(data_modulated, data_vector)
    #calculate mean and standard deviation of number of place like fields
    n_place = [len(d[1]) for d in data_place]
    avg_place_fields = np.mean(n_place)
    std_place_fields = np.std(n_place)
    
    #calculate mean and standard deviation of number of vector fields
    n_vector = [len(d[1]) for d in data_vector]
    avg_vector_fields = np.mean(n_vector)
    std_vector_fields = np.std(n_vector)
    
    #calculate mean and standard deviation of number of modulated fields
    n_mod = [len(d[1]) for d in data_modulated]
    avg_mod_fields = np.mean(n_mod)
    std_mod_fields = np.std(n_mod)
    
    #calculate mean and standard deviation of number of empty fields
    n_empty_all = [np.size(d[0]) for d in data_empty]
    avg_empty_fields = np.mean(n_empty_all)
    std_empty_fields = np.std(n_empty_all)
    
    #calculate mean and standard deviation of number of partially empty fields
    n_empty_some = [np.size(d[1]) for d in data_empty]
    avg_pempty_fields = np.mean(n_empty_some)
    std_pempty_fields = np.std(n_empty_some)
    
    n_other = [total_fields - (n_place[i] + n_empty_all[i] +
                     n_empty_some[i] + n_mod[i] + n_vector[i]) for i in range(len(n_place))]
    avg_other_fields = np.mean(n_other)
    std_other_fields = np.std(n_other)
    
    avg = [avg_place_fields, avg_vector_fields, avg_mod_fields, avg_empty_fields,
           avg_pempty_fields, avg_other_fields]
    std = [std_place_fields, std_vector_fields, std_mod_fields, std_empty_fields,
           std_pempty_fields, std_other_fields]
    
    return avg, std

if __name__=='__main__':

    locs = ["/../time_evolution/guidance_allocentric/",
            "/../time_evolution/guidance_egocentric/",
            "/../time_evolution/aiming_allocentric/",
            "/../time_evolution/aiming_egocentric/"]
    trials = np.arange(0,4200,200)
    savename = ['guidance_allo_time_evolution.npy',
                'guidance_ego_time_evolution.npy',
                'aiming_allo_time_evolution.npy',
                'aiming_ego_time_evolution.npy']
    for i,loc in enumerate(locs[:-2]) :
        avg_place = []
        std_place = []
        avg_vec = []
        std_vec = []
        avg_mod = []
        std_mod = []
        avg_emp = []
        std_emp = []
        avg_pemp = []
        std_pemp = []
        avg_oth = []
        std_oth = []
        for trial in trials : 
            avg, std = avg_std_field_types(loc, trial, n_units)
            
            avg_place.append(avg[0])
            avg_vec.append(avg[1])
            avg_mod.append(avg[2])
            avg_emp.append(avg[3])
            avg_pemp.append(avg[4])
            avg_oth.append(avg[5])
            
            std_place.append(std[0])
            std_vec.append(std[1])
            std_mod.append(std[2])
            std_emp.append(std[3])
            std_pemp.append(std[4])
            std_oth.append(std[5])
    
        avg_place = np.array(avg_place)
        avg_vec = np.array(avg_vec)
        avg_mod = np.array(avg_mod)
        avg_emp = np.array(avg_emp)
        avg_pemp = np.array(avg_pemp)
        avg_oth = np.array(avg_oth)
        all_avg = np.vstack((avg_place,avg_vec,avg_mod,
                              avg_emp,avg_pemp,avg_oth))
        all_std = np.vstack((std_place,std_vec,std_mod,
                             std_emp,std_pemp,std_oth))
        data = {'AVG' : all_avg, 'STD' : all_std}
        np.save(savename[i],data)

