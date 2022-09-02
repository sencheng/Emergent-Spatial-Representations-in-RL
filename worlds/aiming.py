# basic imports
import blender_frontend

class FrontendAiming(blender_frontend.BlenderFrontend) : 
    
    def __init__(self, control_buffer_size=1000):
        
        super().__init__(control_buffer_size=control_buffer_size)
        self.functions['teleportXY'] = self.teleportXY
    
    def teleportXY(self, data) :
        objectName, xStr, yStr = data
        x=float(xStr)
        y=float(yStr)
        object=self.scene.objects[objectName]
        object.worldPosition.x = x
        object.worldPosition.y = y
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
BF = FrontendAiming()
BF.main_loop()