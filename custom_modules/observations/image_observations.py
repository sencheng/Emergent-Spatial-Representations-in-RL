from cobel.observations.image_observations import ImageObservationBaseline
# basic imports
import numpy as np
from PyQt5.QtCore import QRectF
# open-cv
import cv2

def add_gaussian_noise(observation, mean, var) :    
    row, col, ch= observation.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return observation + gauss
    
def crop_image(image, image_dim, view_angle) : 
    keep_pixels = image_dim[0] * view_angle / 360.0
    remove_pixels = image_dim[0] - keep_pixels
    rm = np.ceil(remove_pixels/2).astype('int32')
    cropped_image = image[:,rm:rm+np.ceil(keep_pixels).astype('int32')]
    return cropped_image
    

class ImageObservationFOV(ImageObservationBaseline) : 
    
    def __init__(self, world, guiParent, visualOutput=True, imageDims=(30, 1),
                 view_angle=360.0, noise=(0.0,0.0)) :
        super().__init__(world, guiParent, visualOutput=visualOutput, imageDims=imageDims)
        self.view_angle = view_angle 
        self.observation = crop_image(image=np.zeros((self.imageDims[1],
                                                      self.imageDims[0],3)),
                                         image_dim=imageDims, 
                                         view_angle=self.view_angle)
        self.noise = noise
        
        # _observe determines if the observation is actually recorded from the
        #environment (true observation) or replaced by dummy data.
        #This should normally always be set to True, but can be used to temporarily
        #turn off the observation to simulate the loss of sensory signal
        self._observe = True
        
    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        '''
        # the observation is plainly the robot's camera image data
        if self._observe : observation = self.worldModule.envData['imageData']
        else : observation = np.zeros((self.imageDims[1], self.imageDims[0],3))
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.cameraImage.setImage(imageData)
            imageScale = 1.0
            self.cameraImage.setRect(QRectF(0.0, 0.0, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.imageDims)
        #resize according to field of view
        observation = crop_image(observation, self.imageDims, self.view_angle)
        observation.astype('float32')  
        observation = add_gaussian_noise(observation, self.noise[0], self.noise[1])
        observation = observation/255.0
        # display the observation camera image reduced to one line
        if self.visualOutput:
            imageData = observation
            self.observationImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.observationImage.setImage(imageData)
            imageScale = 1.0
            self.observationImage.setRect(QRectF(0.0, -0.1, imageScale, 
                                                 imageData.shape[0]/imageData.shape[1]*imageScale))
        self.observation = observation
        
    def set_observation_state(self, state) :
        self._observe = state