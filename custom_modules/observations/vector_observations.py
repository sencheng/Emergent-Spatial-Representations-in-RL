import numpy as np
import gym

def calculate_angle_ref(x1, y1, x2, y2, ref=(1,0)) : 

    #translate to have tail at (0,0)
    current_vector = (x2 - x1, y2 - y1)
    angle = np.arctan2(current_vector[1], current_vector[0]) - np.arctan2(ref[1],ref[0])
    if angle >= 0.0 : 
        angle = np.rad2deg(angle)
    else : 
        angle = 360.0 + np.rad2deg(angle)
    return angle

def add_gaussian_noise(observation, mean, var) :    
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(observation.shape))
    gauss = gauss.reshape(observation.shape)
    return observation + gauss

class VectorObservation() : 
    
    def __init__(self, topology, gui_parent, vector_encoding='egocentric',
                 noise=(0.0,0.0)) :
        
        encoding_list = ['egocentric', 'allocentric']
        if vector_encoding not in encoding_list : 
            raise ValueError("Vector encoding must be either 'egocentric' or 'allocentric'")
        
        self.topology        = topology
        self.gui_parent      = gui_parent
        self.vector_encoding = vector_encoding
        self.noise = noise
    
        if self.vector_encoding == 'egocentric' :
            self.observation     = np.array([0.0,0.0])
        elif self.vector_encoding == 'allocentric' :
            self.observation     = np.array([0.0,0.0,0.0]) 
            
        # _observe determines if the observation is actually recorded from the
        #environment (true observation) or replaced by dummy data.
        #This should normally always be set to True, but can be used to temporarily
        #turn off the observation to simulate the loss of sensory signal
        self._observe = True
        
    def add_topology_graph(self, topology_graph) : 
        self.topology = topology_graph
    
    def update(self) : 
        '''
        Updates the observation. For observing a vector to the goal, this will
        compute the vector between the current node on the topology graph and
        the goal node.

        Returns
        -------
        None.

        '''
        current_node  = self.topology.current_node
        goal_nodes    = self.topology.goal_nodes
        
        current_x = self.topology.nodes[current_node].x
        current_y = self.topology.nodes[current_node].y
        
        #For now, we'll assume a single goal node is present. 
        #This code will need to be modified to accomodate multiple goal
        #locations
        
        goal_x = self.topology.nodes[goal_nodes[0]].x
        goal_y = self.topology.nodes[goal_nodes[0]].y
        
        distance   = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        #this is the angle of the line connecting the goal and current location
        #w.r.t a reference vector pointing due east
        allocentric_angle = calculate_angle_ref(current_x, current_y, goal_x, goal_y)
        #this is the current heading direction of the agent
        head_direction    = self.topology.head_direction
        #egocentric direction - we calculate the direction of the goal wrt the
        #heading direction
        
        hd_rad = np.deg2rad(head_direction)
        
        egocentric_angle = calculate_angle_ref(current_x,current_y, goal_x, goal_y,
                                           ref=(np.cos(hd_rad), np.sin(hd_rad)))
        egocentric_angle = egocentric_angle / 360.0
        max_distance = np.sqrt((self.topology.nodes[0].x - self.topology.nodes[-1].x)**2 + (self.topology.nodes[0].y - self.topology.nodes[-1].y)**2)
        distance = distance / max_distance #normalize distance
        
        if self.vector_encoding=='egocentric' : 
            if self._observe : 
                observation = np.array([distance, egocentric_angle])
                observation = add_gaussian_noise(observation, self.noise[0], self.noise[1])
            else : observation = np.array([0.0,0.0])            
            self.observation = observation
            
        if self.vector_encoding=='allocentric' : 
            if self._observe : 
                observation = np.array([distance, allocentric_angle,
                                             head_direction])
                observation = add_gaussian_noise(observation, self.noise[0], self.noise[1])
            else : np.array([0.0,0.0,0.0]) 
            self.observation = observation
        
    def set_observation_state(self, state) :
        self._observe = state
    
    def getObservationSpace(self):
        
        '''
        This function returns the observation space for the given observation class.
        '''
        return gym.spaces.Box (low=0.0, high=1.0, shape=(2,))