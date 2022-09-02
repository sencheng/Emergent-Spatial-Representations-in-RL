import numpy as np
from cobel.spatial_representations.topology_graphs.simple_topology_graph  import HexagonalGraph
import gym
import random

class HexagonalGraphAllocentric(HexagonalGraph) :
    
    def generate_behavior_from_action(self, action) :
        
        callback_value = dict()
        trans_actions = np.arange(0,6)
        rot_actions   = np.arange(6,12)
        orientations = np.arange(0,360,60)
        hd_dict = dict(zip(rot_actions, orientations))
        
        if action!='reset' : 
            if action in trans_actions : 
                node_id = self.nodes[self.current_node].neighbors[action].index
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)                  
                else  :
                    self.next_node = self.current_node
                    self.world_module.goalReached = True
                    
            if action in rot_actions : 
                node_id = self.current_node
                self.head_direction = hd_dict[action]
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)
                                
        else : 
            node_id = random.choice(self.start_nodes)
            directions = self.calculate_angles_edges(self.nodes[node_id],[0,1])[0]
            random_direction = random.choice(directions)
            self.head_direction = random_direction[2]
            self.move_to_node(node_id, random_direction[2])
        
        self.current_node = self.next_node
        callback_value['currentNode'] = self.nodes[self.current_node]
        
        return callback_value
    
    def get_action_space(self) :
        
        return gym.spaces.Discrete(12)
    
class HexagonalGraphExtended(HexagonalGraph) : 
    
    def generate_behavior_from_action(self, action) :
        
        callback_value = dict()
        #allocentric actions
        trans_actions = np.arange(0,6)
        rot_actions   = np.arange(6,12)
        orientations = np.arange(0,360,60)
        hd_dict = dict(zip(rot_actions, orientations))
        #egocentric copies
        forward_actions = [12,15,18,21]
        left_actions = [13,16,19,22]
        right_actions = [14,17,20,23]
        
        if action!='reset' : 
            
            heading = np.array([self.world_module.envData['poseData'][2],
                                self.world_module.envData['poseData'][3]])
            
            heading = heading/np.linalg.norm(heading)
            node = self.nodes[self.current_node]

            forward_edge, left_edges, right_edges = self.calculate_angles_edges(node,
                                                                                heading,
                                                                                threshold=5)[1:]
            if action in trans_actions : 
                node_id = self.nodes[self.current_node].neighbors[action].index
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)                    
                else  :
                    self.next_node = self.current_node
                    self.world_module.goalReached = True
            
            if action in rot_actions : 
                node_id = self.current_node
                self.head_direction = hd_dict[action]
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)
            
            if action in forward_actions : 
                angle = 180.0 / np.pi * np.arctan2(heading[1], heading[0])
                self.head_direction = angle
                if len(forward_edge) != 0 :
                    next_node_id = forward_edge[0]
                    self.move_to_node(next_node_id, angle)
                else : 
                    self.next_node = self.current_node

            if action in left_actions : 
                angle = 180.0 / np.pi * np.arctan2(left_edges[0][1][1],
                                                 left_edges[0][1][0])
                self.head_direction = angle
                next_node_id = self.current_node
                self.move_to_node(next_node_id, angle)

            if action in right_actions : 
                angle = 180.0 / np.pi * np.arctan2(right_edges[0][1][1],
                                                 right_edges[0][1][0])
                self.head_direction = angle
                next_node_id = self.current_node
                self.move_to_node(next_node_id, angle)

        else : 
            node_id = random.choice(self.start_nodes)
            directions = self.calculate_angles_edges(self.nodes[node_id],[0,1])[0]
            random_direction = random.choice(directions)
            self.head_direction = random_direction[2]    

        
        self.current_node = self.next_node
        callback_value['currentNode'] = self.nodes[self.current_node]
        
        return callback_value
    
    def get_action_space(self) :
        
        return gym.spaces.Discrete(24)