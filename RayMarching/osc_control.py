import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server

class OscControl():
    
    def __init__(self, skeleton, visualization, address, port):
        
        self.skeleton = skeleton
        self.visualization = visualization
        self.address = address 
        self.port = port
         
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/updatesmoothing", self.setMocapUpdateSmoothing)
        self.dispatcher.map("/mocap/skelposworld", self.setMocapSkeletonPosition)
        self.dispatcher.map("/mocap/joint/pos_world", self.setMocapJointPositions)
        self.dispatcher.map("/mocap/0/joint/pos_world", self.setMocapJointPositions)
        self.dispatcher.map("/mocap/joint/rot_world", self.setMocapJointRotations)
        self.dispatcher.map("/mocap/0/joint/rot_world", self.setMocapJointRotations)
        
        self.dispatcher.map("/vis/camposition", self.setVisCamPosition)
        self.dispatcher.map("/vis/camangle", self.setVisCamAngle)
        
        self.dispatcher.map("/vis/bgcolor", self.setVisBGColor)
        self.dispatcher.map("/vis/objectcolor", self.setVisObjectColor)
        
        self.dispatcher.map("/vis/lightposition", self.setVisLightPosition)
        self.dispatcher.map("/vis/lightambientscale", self.setVisLightAmbientScale)
        self.dispatcher.map("/vis/lightdiffusescale", self.setVisLightDiffuseScale)
        self.dispatcher.map("/vis/lightspecularscale", self.setVisLightSpecularScale)
        self.dispatcher.map("/vis/lightspecularpow", self.setVisLightSpecularPow)
        
        self.dispatcher.map("/vis/lightocclusionscale", self.setVisLightOcclusionScale)
        self.dispatcher.map("/vis/lightocclusionrange", self.setVisLightOcclusionRange)
        self.dispatcher.map("/vis/lightocclusionresolution", self.setVisLightOcclusinResolution)
        
        self.dispatcher.map("/vis/jointprimitive", self.setVisJointPrimitive)
        self.dispatcher.map("/vis/jointsize", self.setVisJointSize)
        self.dispatcher.map("/vis/jointround", self.setVisJointRounding)
        self.dispatcher.map("/vis/jointsmooth", self.setVisJointSmoothing)
        
        self.dispatcher.map("/vis/edgeprimitive", self.setVisEdgePrimitive)
        self.dispatcher.map("/vis/edgesize", self.setVisEdgeSize)
        self.dispatcher.map("/vis/edgeround", self.setVisEdgeRounding)
        self.dispatcher.map("/vis/edgesmooth", self.setVisEdgeSmoothing)
        
        self.dispatcher.map("/vis/jointedgesmooth", self.setVisJointEdgeSmoothing)
        
        self.dispatcher.map("/vis/groundprimitive", self.setVisGroundPrimitive)
        self.dispatcher.map("/vis/groundposition", self.setVisGroundPosition)
        self.dispatcher.map("/vis/groundrotation", self.setVisGroundRotation)
        self.dispatcher.map("/vis/groundsize", self.setVisGroundSize)
        self.dispatcher.map("/vis/groundround", self.setVisGroundRounding)
        self.dispatcher.map("/vis/groundsmooth", self.setVisGroundSmoothing)
    
    
        self.server = osc_server.ThreadingOSCUDPServer((self.address, self.port), self.dispatcher)
        
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
    
    def setMocapUpdateSmoothing(self, address, *args):
        
        self.skeleton.setUpdateSmoothing(args[0])
        
    def setMocapSkeletonPosition(self, address, *args):
        
        position = np.array(args)

        self.skeleton.setPosition(position)
        
    def setMocapJointPositions(self, address, *args):

        argCount = len(args)
        posCount = argCount // 3
        
        positions = np.array(args)
        positions = np.reshape(positions, (-1, 3))
        
        # right handed to left handed
        tmp = np.copy(positions)
        positions[:, 0] = tmp[:, 1]
        positions[:, 1] = tmp[:, 0]
        positions[:, 2] = tmp[:, 2]
        
        self.skeleton.setJointPositions(positions)

    def setMocapJointRotations(self, address, *args):

        argCount = len(args)
        rotCount = argCount // 4
        
        rotations = np.array(args)
        rotations = np.reshape(rotations, (-1, 4))

        tmp = np.copy(rotations)
        rotations[:, 0] = tmp[:, 0]
        rotations[:, 1] = tmp[:, 2]
        rotations[:, 2] = tmp[:, 1]
        rotations[:, 3] = tmp[:, 3]
        
        self.skeleton.setJointRotations(rotations)
        
    def setVisCamPosition(self, address, *args):
        
        camPosition = np.array(args)

        self.visualization.setCamPosition(camPosition)

    def setVisCamAngle(self, address, *args):
        
        camAngle = np.array(args)

        self.visualization.setCamAngle(camAngle)
        
    def setVisBGColor(self, address, *args):

        bgColor = np.array(args)
        
        self.visualization.setBGColor(bgColor)
        
    def setVisObjectColor(self, address, *args):
        
        objectColor = np.array(args)
        
        self.visualization.setObjectColor(objectColor)
        
    def setVisLightPosition(self, address, *args):
        
        lightPosition = np.array(args)
        
        self.visualization.setLightPosition(lightPosition)
        
    def setVisLightAmbientScale(self, address, *args):
        
        scale = args[0]

        self.visualization.setLightAmbientScale(scale)
        
    def setVisLightDiffuseScale(self, address, *args):
        
        scale = args[0]
        
        self.visualization.setLightDiffuseScale(scale)
        
    def setVisLightSpecularScale(self, address, *args):
        
        scale = args[0]
        
        self.visualization.setLightSpecularScale(scale)
        
    def setVisLightSpecularPow(self, address, *args):
        
        pow = args[0]
        
        self.visualization.setLightSpecularPow(pow)

    def setVisLightOcclusionScale(self, address, *args):
        
        scale = args[0]
        
        self.visualization.setLightOcclusionScale(scale)

    def setVisLightOcclusionRange(self, address, *args):
        
        range = args[0]
        
        self.visualization.setLightOcclusionRange(range)
        
    def setVisLightOcclusinResolution(self, address, *args):
        
        resolution = args[0]
        
        self.visualization.setLightOcclusinResolution(resolution)
        
    def setVisJointPrimitive(self, address, *args):
        
        if len(args) == 1:
            
            primitive = args[0]
            
            self.visualization.setJointPrimitives(primitive)
            
        elif len(args) == 2:
            
            index = args[0]
            primitive = args[1]
            
            self.visualization.setJointPrimitive(index, primitive)
            
    def setVisJointSize(self, address, *args):
        
        if len(args) == 3:
            
            size = np.array(args)
            
            self.visualization.setJointSizes(size)
            
        elif len(args) == 4:
            
            index = args[0]
            size = np.array(args[1:])
            
            self.visualization.setJointSize(index, size)

    def setVisJointRounding(self, address, *args):
        
        if len(args) == 1:
            
            round = args[0]
            
            self.visualization.setJointRoundings(round)
            
        elif len(args) == 2:
            
            index = args[0]
            round = args[1]
            
            self.visualization.setJointRounding(index, round)

    def setVisJointSmoothing(self, address, *args):
        
        if len(args) == 1:
            
            smooth = args[0]
            
            self.visualization.setJointSmoothings(smooth)
            
        elif len(args) == 2:
            
            index = args[0]
            smooth = args[1]
            
            self.visualization.setJointSmoothing(index, smooth)

            
    def setVisEdgePrimitive(self, address, *args):

        if len(args) == 1:
            
            primitive = args[0]
            
            self.visualization.setEdgePrimitives(primitive)
            
        elif len(args) == 2:
            
            index = args[0]
            primitive = args[1]
            
            self.visualization.setEdgePrimitive(index, primitive)

    def setVisEdgeSize(self, address, *args):
        
        if len(args) == 3:
            
            size = np.array(args)
            
            self.visualization.setEdgeSizes(size)
            
        elif len(args) == 4:
            
            index = args[0]
            size = np.array(args[1:])
            
            self.visualization.setEdgeSize(index, size)

    def setVisEdgeRounding(self, address, *args):
        
        if len(args) == 1:
            
            round = args[0]
            
            self.visualization.setEdgeRoundings(round)
            
        elif len(args) == 2:
            
            index = args[0]
            round = args[1]
            
            self.visualization.setEdgeRounding(index, round)

    def setVisEdgeSmoothing(self, address, *args):
        
        if len(args) == 1:
            
            smooth = args[0]
            
            self.visualization.setEdgeSmoothings(smooth)
            
        elif len(args) == 2:
            
            index = args[0]
            smooth = args[1]
            
            self.visualization.setEdgeSmoothing(index, smooth)
            
    def setVisJointEdgeSmoothing(self, address, *args):
        
        smooth = args[0]
        
        self.visualization.setJointEdgeSmoothing(smooth)
        
    def setVisGroundPrimitive(self, address, *args):

        primitive = args[0]
        
        self.visualization.setGroundPrimitive(primitive)

    def setVisGroundPosition(self, address, *args):

        position = np.array(args)
        
        self.visualization.setGroundPosition(position)

    def setVisGroundRotation(self, address, *args):

        rotation = np.array(args)
        
        self.visualization.setGroundRotation(rotation)

    def setVisGroundSize(self, address, *args):
        
        size = np.array(args)
            
        self.visualization.setGroundSize(size)

    def setVisGroundRounding(self, address, *args):

        rounding = args[0]
        
        self.visualization.setGroundRounding(rounding) 
        
    def setVisGroundSmoothing(self, address, *args):

        smoothing = args[0]
        
        self.visualization.setGroundSmoothing(smoothing)      
        
