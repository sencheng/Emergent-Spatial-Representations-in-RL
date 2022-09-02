from cobel.frontends.frontends_blender import FrontendBlenderInterface

class BlenderOnlineRenderer(FrontendBlenderInterface) :
    
    def teleportXY(self, objectName, x, y) : 
        xy ='%f,%f'%(x,y)
        sendStr = 'teleportXY,%s,%s'%(objectName,xy)
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)
