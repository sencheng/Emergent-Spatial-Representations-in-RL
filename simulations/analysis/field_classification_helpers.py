"""
A collection of helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.spatial.distance import pdist


def classify_modulated_vector(idx_modulated, idx_vector) :
    """ Using classification algorithms for HD modulated fields and ECD fields
    will return redundant results. This function can be used to remove 
    redundant entries. 
    
    Keyword arguments :
    idx_modulated -- list of unit ids that are HD modulated
    idx_vector -- list of unit ids that are ECD like
    """
    mask         = np.in1d(idx_modulated, idx_vector)
    mask2        = np.in1d(idx_vector, idx_modulated)
    id_modulated = idx_modulated[~mask]
    id_vector    = idx_modulated[mask]
    return mask, mask2, id_modulated, id_vector


def reclassify(data_modulated, data_vector) :
    """ Re-classifies data from the classification of HD modulated and ECD fields.
    The reclassification is done in place.
    
    Keyword arguments : 
    data_modulated : list of modulated fields in the form 
    (fields, field_ids, pairwise_distances_centers, pairwise_distances_hd)
    
    """
    for m,v in zip(data_modulated,data_vector) :
        mask, mask2, ids_mod, ids_vec = classify_modulated_vector(m[1], v[1])
        m[1] = ids_mod
        v[1] = ids_vec
        m[0] = (m[0])[~mask]
        v[0] = (v[0])[mask2]
        
def classify_place_field(activity_map) :
    """ Decide whether or not a given spatial activity map contains place fields.
    
    Keyword arguments : 
    activity_map : 2d numpy array of spatial activations

    """
    is_field = True
    #find clusters where activity falls off to 15%
    #and calculate a boolean mask of clusters
    activity_map = nd.gaussian_filter(activity_map,sigma=2)
    activity_map = np.where(activity_map<0.15*np.max(activity_map),0,
                            activity_map)
    mask = activity_map > 0
    
    #label clusters
    labels, nb = nd.label(mask)

    if nb != 0 :
        #find largest cluster
        sizes = []
        for i in range(nb+1) :
            slice_id = nd.find_objects(labels==i)
            #sizes.append(np.shape(activity_map[slice_id[0]]))
            if not slice_id :
                sizes.append((25,25))
            else : sizes.append(np.shape(activity_map[slice_id[0]]))
        pixels = np.prod(sizes, axis=1)
        l = np.argmax(pixels[1:]) + 1
        
        field = nd.gaussian_filter(np.where(labels==l,activity_map,0),sigma=2)
    
    else :
        field = labels
    #is it a field?
    if nb == 0 :
        is_field=False
        print("no cluster -- not place like")
    elif nb>4 : 
        is_field=False
        print("too many -- not place like")
    elif pixels[l] > 0.50*pixels[0] :
        non_zero = np.count_nonzero(field)
        fraction_active = non_zero/625
        is_field=False
        print("too large, not place like : ", fraction_active)
    return is_field, field

def is_empty(activity_map) :
    """ Stub for deciding whether a spatial activity map is completely empty."""
    is_empty = False
    if np.all(activity_map == 0) :
        is_empty = True
    return is_empty

def empty_fields(file) :
    """ Returns all unit ids that show no spatial activity in all head directions,
    and in any of the head directions.
    """
    fields = np.squeeze(np.array(np.load(file).item()['FIELDS']))
    n_units = fields.shape[2]
    empty = np.zeros((6,n_units))
    for f,i in zip(np.rollaxis(fields,2),range(n_units)) :
        for hd,j in zip(f,range(6)) :
            hd = hd.reshape(25,25)
            empty[j,i] = is_empty(hd)
    a = np.all(empty, axis=0)        
    ids_all = np.squeeze(np.array(np.where(a)))
    b = np.any(empty, axis=0)
    ids_any = np.squeeze(np.array(np.where(b)))
    ids_any = np.setdiff1d(ids_any, ids_all)
    return ids_all, ids_any

def calculate_pdist(fields_sliced, place_indices) :
    """ Calculate pairwise distances between field centers for specified fields.
    
    Keyword arguments :
    fields_sliced : fields for which the pairwise distances must be calculated
    place_indices : (legacy, remove) #TODO
    """
    centers = np.zeros((len(place_indices),6,2))
    for f,i in zip(fields_sliced,range(len(fields_sliced))) :
        for angle,j in zip(f,range(len(f))) :
            centers[i,j,:] = divmod((np.argmax(angle)),25)
    pdist_centers = [pdist(centers[i]) for i in range(len(centers))]
    angles    = np.deg2rad(np.arange(0,360,60,dtype='int32').reshape(-1,1))
    vectors   = np.squeeze(np.array([(np.cos(a),np.sin(a)) for a in angles]))
    return pdist_centers, pdist(vectors)

def plot_field(fields, show_centers=True) :

    fig,ax = plt.subplots(fields.shape[0],fields.shape[1],
                          figsize=(fields.shape[1],fields.shape[0]))
    centers = np.zeros((len(fields),6,2))
    for f,i in zip(fields,range(len(fields))) :
        for angle,j in zip(f,range(len(f))) :
            centers[i,j,:] = divmod((np.argmax(angle)),25)
            angle = angle.reshape(25,25)

            ax[i,j].imshow(nd.gaussian_filter(angle,sigma=2),cmap='jet',
              origin='lower')
            if show_centers :
                ax[i,j].scatter([centers[i,j,1]],[centers[i,j,0]],c='white',
                  s=50)
            ax[i,j].axis('off')
            
def load_fields(file, indices) : 
    fields = np.squeeze(np.array(np.load(file).item()['FIELDS']))
    fields_sliced = np.array([fields[:,:,p] for p in indices])
    return fields_sliced

def place_like(fields) : 
    fields = np.squeeze(np.array(np.load(fields).item()['FIELDS']))

    n_units = fields.shape[2]
    mean_fields = np.mean(fields,axis=0).reshape((1,625,n_units))
    stacked = np.vstack((fields,mean_fields))
    
    is_field = np.zeros((7,n_units))
    largest_field  = np.zeros((7,625,n_units))
    print("Place")
    for f,i in zip(np.rollaxis(stacked,2),range(n_units)) :
        print("Unit",i)
        for hd,j in zip(f,range(7)) :
            print("Head direction",j)
            hd = hd.reshape(25,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest_field[j,:,i] = big.flatten()
            
    #if all fields AND mean are classified, it may be a place like representation
    
    a = np.all(is_field, axis=0)
    ids = np.squeeze(np.array(np.where(a))) #ids where fields are place like
    if ids.shape != () : 
        f = np.array([largest_field[:,:,p] for p in ids])
    else :
        f = np.array([largest_field[:,:,ids]])
    
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd =  [15*np.ones((15,))]
        v =  [15*np.ones((15,))]
    #if there is too much directional modulation, not field
    x_all = np.all(np.array(pd) < 10, axis=1)
    ids = np.where(x_all,ids,-1)
    ids = ids[ids != -1] 
    f = np.array([largest_field[:,:,p] for p in ids])
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd =  [15*np.ones((15,))]
    
    return f,ids,pd,v

def vector_cells(file) :
    fields = np.squeeze(np.array(np.load(file).item()['FIELDS']))
    n_units = fields.shape[2]
    new_fields = np.zeros((fields.shape))
    
    for f,i in zip(np.rollaxis(fields,2),range(n_units)) :
        for hd,j in zip(f, range(6)) :
            hd = hd.reshape(25,25)
            angles = [90, 30, 330, 270, 210, 150]
            hd = nd.rotate(hd, 90.0-angles[j], reshape=False, cval=np.min(hd))
            new_field = hd.flatten()
            new_fields[j,:,i] = new_field
     
    mean_fields = np.mean(new_fields, axis=0).reshape((1,625,n_units))
    stacked = np.vstack((new_fields,mean_fields))

    is_field = np.zeros((7,n_units))
    largest_field  = np.zeros((7,625,n_units))
    
    for f,i in zip(np.rollaxis(stacked,2),range(n_units)) :
        for hd,j in zip(f,range(7)) :
            hd = hd.reshape(25,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest_field[j,:,i] = big.flatten()
            
    #if all fields AND mean are classified, it may be a place like representation
    a = np.all(is_field, axis=0)
    ids = np.array(np.where(a))
    ids = ids.squeeze(axis=0)

    f = np.array([largest_field[:,:,p] for p in ids])
    if ids.size > 0 : 
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else : 
        pd = [15*np.ones((15,))]
        print(pd)
        v = [15*np.ones((15,))]
    #if there is too much directional modulation, not field
    x_all = np.all(np.array(pd) < 10, axis=1)
    ids = np.where(x_all,ids,-1)
    ids = ids[ids != -1] 
    f = np.array([largest_field[:,:,p] for p in ids])
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd = [15*np.ones((15,))]
        print(pd)
    return f,ids,pd,v
    
    

def modulated(fields) :
    fields = np.squeeze(np.array(np.load(fields).item()['FIELDS']))
    n_units = fields.shape[2]
    is_field = np.zeros((6,n_units))
    largest_field = np.zeros((6,625,n_units))
    
    for f,i in zip(np.rollaxis(fields,2), range(n_units)) :
        for hd, j in zip(f, range(6)) :
            hd = hd.reshape(25,25)

            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest_field[j,:,i] = big.flatten()
            
    a = np.all(is_field, axis=0)
    ids = np.squeeze(np.array(np.where(a)))
    if ids.shape != () : 
        f = np.array([largest_field[:,:,p] for p in ids])
    else : 
        f = np.array([largest_field[:,:,ids]])
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        pd = [np.zeros((15,))]
        v  = [np.zeros((15,))]
    x_any = np.any(np.array(pd) >= 10, axis=1)
    ids = np.where(x_any,ids,-1)
    ids = ids[ids != -1] 
    
    f = np.array([largest_field[:,:,p] for p in ids])
    if ids.size > 0 :
        pd,v = calculate_pdist(f[:,:6,:],ids)
    else :
        print("no slice found--",ids )
        pd = [np.zeros((15,))]
    return f,ids,pd,v

def largest_fields(file) : 
    fields = np.squeeze(np.array(np.load(file).item()['FIELDS']))
    is_field = np.zeros((6,25))
    largest  = np.zeros((6,625,25))
    for f,i in zip(np.rollaxis(fields,2),range(25)) :
        for hd,j in zip(f,range(6)) :
            hd = hd.reshape(25,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest[j,:,i] = big.flatten()
            
    return largest, is_field

def largest_fields_2(fields,n) : 
    is_field = np.zeros((6,n))
    largest  = np.zeros((6,625,n))
    for f,i in zip(np.rollaxis(fields,2),range(n)) :
        for hd,j in zip(f,range(6)) :
            hd = hd.reshape(25,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest[j,:,i] = big.flatten()
            
    return largest, is_field

def largest_fields_linear(fields,n) : 
    is_field = np.zeros((1,n))
    largest  = np.zeros((1,200,n))
    for f,i in zip(np.rollaxis(fields,2),range(n)) :
        for hd,j in zip(f,range(1)) :
            hd = hd.reshape(8,25)
            isit, big = classify_place_field(hd)
            is_field[j,i] = isit
            largest[j,:,i] = big.flatten()
            
    return largest, is_field





