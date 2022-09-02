import numpy as np
from field_classification_helpers import place_like, modulated, empty_fields, vector_cells
import os
import pandas as pd

def stats(loc) :

    files    = os.listdir(loc)
    file_fields   = [f for f in files if f.startswith('fields') and f.endswith('.npy')]
    file_fields.sort()

    data_place = []
    data_empty = []
    data_modulated = []
    data_vector = []
    for f in file_fields : 
        fields,ids,pd,v = place_like(loc+f)
        data_place.append([fields, ids, pd, v])
        ids_all, ids_any = empty_fields(loc+f)
        data_empty.append([ids_all, ids_any])
        fls, i, pa, va = modulated(loc+f)
        data_modulated.append([fls, i, pa, va])
        fs,idxs,ps,vs = vector_cells(loc+f)
        data_vector.append([fs, idxs, ps, vs])
    
    return data_place, data_empty, data_modulated, data_vector

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


place= []
empty_all = []
empty_some = []
hd_modulate = []
vector = []
other = []

ids_data = []

# =============================================================================

pref = "../full_model/"
folders = ["data_guidance/",
           "data_aiming/"]

for f in folders :
    data_place, data_empty, data_modulated, data_vector = stats(pref+f)
    reclassify(data_modulated, data_vector)
    
    n_place = [np.size(d[1]) for d in data_place]
    place.append(n_place)
    
    n_empty_all = [np.size(d[0]) for d in data_empty]
    empty_all.append(n_empty_all)
    
    n_empty_some = [np.size(d[1]) for d in data_empty]
    empty_some.append(n_empty_some)
    
    n_mod = [np.size(d[1]) for d in data_modulated]
    hd_modulate.append(n_mod)
    
    n_vector = [np.size(d[1]) for d in data_vector]
    vector.append(n_vector)
    
    n_other = [50 - (n_place[i] + n_empty_all[i] +
                     n_empty_some[i] + n_mod[i] + n_vector[i]) for i in range(len(n_place))]
    other.append(n_other)
    
df = pd.DataFrame(columns=['Task','Type','Number'])    

for idx in range(len(place[0])) : 
    df2 = pd.DataFrame({'Task':['Guidance']*6,
    'Type':['Place','Vector-like','HD modulated',
            'View selective','No response','Other'],
    'Number': [place[0][idx],vector[0][idx],hd_modulate[0][idx],
               empty_some[0][idx],empty_all[0][idx],other[0][idx]]})
    df = df.append(df2, ignore_index=True)

for idx in range(len(place[1])) : 
    df2 = pd.DataFrame({'Task':['Aiming']*6,
    'Type':['Place','Vector-like','HD modulated',
            'View selective','No response','Other'],
    'Number': [place[1][idx],vector[1][idx],hd_modulate[1][idx],
               empty_some[1][idx],empty_all[1][idx],other[1][idx]]})
    df = df.append(df2, ignore_index=True)    

df.to_csv("classified_full_model.csv")