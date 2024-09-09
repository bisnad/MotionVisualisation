import numpy as np
import transforms3d as t3d
import OpenGL.GL as gl
import ctypes
import logging
import time

from skeleton import Skeleton

class Visualization():
    def __init__(self, skeleton, vertexCode, fragmentCode):
        self.skeleton = skeleton
        self.vertexCode = vertexCode
        self.fragmentCode = fragmentCode
        self.resolution = [1280.0, 720.0]
        
        jointCount = skeleton.getJointCount()
        edgeCount = skeleton.getEdgeCount()
        groundCount = 1
        
        self.skelPosition = np.array([0.0, 0.0, 0.0])
        
        self.camPosition = np.array([1.0, 0.0, 0.0])
        self.camAngle = 45.0
        
        self.bgColor = np.array([0.0, 0.0, 0.0])
        self.objectColor = np.array([1.0, 0.0, 0.0])
        
        self.lightPosition = np.array([1.0, 0.0, 0.0])
        self.lightAmbientScale = 0.5
        self.lightDiffuseScale = 0.5
        self.lightSpecularScale = 0.5
        self.lightSpecularPow = 10.0
        
        self.lightOcclusionScale = 1.0
        self.lightOcclusionRange = 3.0
        self.lightOcclusinResolution = 1.0
        
        self.jointPrimitives = np.zeros((jointCount), dtype=np.int32)
        self.jointSizes = np.ones((jointCount, 3)) * 0.1
        self.jointRoundings = np.ones((jointCount)) * 0.01
        self.jointSmoothings = np.ones((jointCount)) * 0.01

        self.edgePrimitives = np.zeros((edgeCount), dtype=np.int32)
        self.edgeSizes = np.ones((edgeCount, 3))
        self.edgeSizes[:, 0] *= 0.01
        self.edgeSizes[:, 1] *= 0.01
        self.edgeSizes[:, 2] *= 1.0
        self.edgeRoundings = np.ones((jointCount)) * 0.1
        self.edgeSmoothings = np.ones((edgeCount)) * 0.01
        self.jointEdgeSmoothing = 0.0
        
        self.groundPrimitive = 0
        self.groundPosition = np.array([0.0, 0.0, 0.0])
        self.groundRotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.groundTransform = np.eye(4)
        self.groundSize = np.ones((3)) * 0.1
        self.groundRounding = 0.01
        self.groundSmoothing = 0.01
        
    def setupShader(self, gl):
        
        # setup shader
        self.program = gl.glCreateProgram()
        self.vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        self.fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        
        logger = logging.getLogger(__name__)

        # Set shaders source
        gl.glShaderSource(self.vertex, self.vertexCode)
        gl.glShaderSource(self.fragment, self.fragmentCode)

        # Compile shaders
        gl.glCompileShader(self.vertex)
        if not gl.glGetShaderiv(self.vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(self.vertex).decode()
            logger.error("Vertex shader compilation error: %s", error)

        gl.glCompileShader(self.fragment)
        if not gl.glGetShaderiv(self.fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(self.fragment).decode()
            print(error)
            raise RuntimeError("Fragment shader compilation error")

        gl.glAttachShader(self.program, self.vertex)
        gl.glAttachShader(self.program, self.fragment)
        gl.glLinkProgram(self.program)

        if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(self.program))
            raise RuntimeError('Linking error')
            
        self.shader_iGlobalTime = gl.glGetUniformLocation(self.program, "iGlobalTime")
        self.shader_iResolution = gl.glGetUniformLocation(self.program, "iResolution")
        self.shader_camPosition = gl.glGetUniformLocation(self.program, "camPosition")
        self.shader_camAngle = gl.glGetUniformLocation(self.program, "camAngle")
        
        self.shader_bgColor = gl.glGetUniformLocation(self.program, "bgColor")
        self.shader_objectColor = gl.glGetUniformLocation(self.program, "objectColor")
        
        self.shader_lightPosition = gl.glGetUniformLocation(self.program, "lightPosition")
        self.shader_lightAmbientScale = gl.glGetUniformLocation(self.program, "lightAmbientScale")
        self.shader_lightDiffuseScale = gl.glGetUniformLocation(self.program, "lightDiffuseScale")
        self.shader_lightSpecularScale = gl.glGetUniformLocation(self.program, "lightSpecularScale")
        self.shader_llightSpecularPow = gl.glGetUniformLocation(self.program, "lightSpecularPow")

        self.shader_lightOcclusionScale = gl.glGetUniformLocation(self.program, "lightOcclusionScale")
        self.shader_lightOcclusionRange = gl.glGetUniformLocation(self.program, "lightOcclusionRange")
        self.shader_lightOcclusionResolution = gl.glGetUniformLocation(self.program, "lightOcclusionResolution")
        
        self.shader_jointEdgeSmoothing = gl.glGetUniformLocation(self.program, "jointEdgeSmoothing")
        

        gl.glDetachShader(self.program, self.vertex)
        gl.glDetachShader(self.program, self.fragment)

        gl.glUseProgram(self.program)
        
        # setup render quad

        # Build data
        data = np.zeros((4, 2), dtype=np.float32)
        # Request a buffer slot from GPU
        buffer = gl.glGenBuffers(1)

        # Make this buffer the default one
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

        stride = data.strides[0]

        offset = ctypes.c_void_p(0)
        loc = gl.glGetAttribLocation(self.program, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

        # drawing quad
        data[...] = [(+1, -1), (+1, +1), (-1, -1), (-1, +1)]

        # Upload CPU data to GPU buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        
        
        self.start_time = time.time() 
    
    def render(self, gl):
        gl.glUseProgram(self.program)
        
        elapsed_time = time.time() - self.start_time

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUniform1f(self.shader_iGlobalTime, elapsed_time)
        gl.glUniform2f(self.shader_iResolution, *self.resolution)
        
        gl.glUniform3f(self.shader_camPosition, *self.camPosition.tolist())
        gl.glUniform1f(self.shader_camAngle, self.camAngle);

        gl.glUniform3f(self.shader_bgColor, *self.bgColor.tolist())
        gl.glUniform3f(self.shader_objectColor, *self.objectColor.tolist())
        gl.glUniform3f(self.shader_lightPosition, *self.lightPosition.tolist())
        
        gl.glUniform1f(self.shader_lightAmbientScale, self.lightAmbientScale)
        gl.glUniform1f(self.shader_lightDiffuseScale, self.lightDiffuseScale)
        gl.glUniform1f(self.shader_lightSpecularScale, self.lightSpecularScale)
        gl.glUniform1f(self.shader_llightSpecularPow, self.lightSpecularPow)

        gl.glUniform1f(self.shader_lightOcclusionScale, self.lightOcclusionScale)
        gl.glUniform1f(self.shader_lightOcclusionRange, self.lightOcclusionRange)
        gl.glUniform1f(self.shader_lightOcclusionResolution, self.lightOcclusinResolution)
        
        gl.glUniform1f(self.shader_jointEdgeSmoothing, self.jointEdgeSmoothing)

        jointCount = self.skeleton.getJointCount()
        edgeCount = self.skeleton.getEdgeCount()
        
        jointTransforms = np.copy(self.skeleton.getJointTransforms())
        edgeTransforms = np.copy(self.skeleton.getEdgeTransforms())
        edgeLengths = np.copy(self.skeleton.getEdgeLengths())

        # joint transforms
        for jI in range(jointCount):
            
            jointTransform = jointTransforms[jI]
            
            uniformName = "jointTransforms[" + str(jI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniformMatrix4fv(uniformLoc, 1, gl.GL_FALSE, jointTransform.tolist ())
            
        # joint primitives
        for jI in range(jointCount):
            
            jointPrimitive = self.jointPrimitives[jI]
            uniformName = "jointPrimitives[" + str(jI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1i(uniformLoc, jointPrimitive)
        
        # joint sizes
        for jI in range(jointCount):
            
            jointSize = self.jointSizes[jI]
            uniformName = "jointSizes[" + str(jI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform3fv(uniformLoc, 1, jointSize.tolist())
            
        # joint rounding
        for jI in range(jointCount):
            
            jointRounding = self.jointRoundings[jI]
            uniformName = "jointRoundings[" + str(jI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1f(uniformLoc, jointRounding)
                
        # joint smooths
        for jI in range(jointCount):
            
            jointSmooth = self.jointSmoothings[jI]
            uniformName = "jointSmoothings[" + str(jI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1f(uniformLoc, jointSmooth)
            
        # edge transforms
        for eI in range(edgeCount):
            
            edgeTransform = edgeTransforms[eI]
            uniformName = "edgeTransforms[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniformMatrix4fv(uniformLoc, 1, gl.GL_FALSE, edgeTransform.tolist ())
            
        # edge primitives
        for eI in range(edgeCount):
            
            edgePrimitive = self.edgePrimitives[eI]
            uniformName = "edgePrimitives[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1i(uniformLoc, edgePrimitive)
            
        # edge lengths
        for eI in range(edgeCount):

            edgeLength = edgeLengths[eI]
            uniformName = "edgeLengths[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1f(uniformLoc, edgeLength)
            
        # edge sizes
        for eI in range(edgeCount):

            edgeSize = self.edgeSizes[eI]
            uniformName = "edgeSizes[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform3fv(uniformLoc, 1, edgeSize.tolist())
 
        # edge rounding
        for eI in range(edgeCount):
            
            edgeRounding = self.edgeRoundings[eI]
            uniformName = "edgeRoundings[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1f(uniformLoc, edgeRounding)           
 
        # edge smooths
        for eI in range(edgeCount):
            
            edgeSmooth = self.edgeSmoothings[eI]
            uniformName = "edgeSmoothings[" + str(eI) + "]";
            uniformLoc = gl.glGetUniformLocation(self.program, uniformName)
            gl.glUniform1f(uniformLoc, edgeSmooth)


        # ground settings
        
        gl.glUniform1i(gl.glGetUniformLocation(self.program, "groundPrimitive"), self.groundPrimitive)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "groundTransform"), 1, gl.GL_FALSE, self.groundTransform.tolist())
        gl.glUniform3fv(gl.glGetUniformLocation(self.program, "groundSize"), 1, self.groundSize.tolist())
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "groundRounding"), self.groundRounding)     
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "groundSmoothing"), self.groundSmoothing)     


        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)        
        
        
    def setBGColor(self, bgColor):
        self.bgColor = bgColor
        
    def setObjectColor(self, objectColor):
        self.objectColor = objectColor
        
    def setCamPosition(self, camPosition):
        self.camPosition = camPosition
        
    def setCamAngle(self, camAngle):
        self.camAngle = camAngle
        
    def setLightPosition(self, position):
        self.lightPosition = position
        
    def setLightAmbientScale(self, scale):
        self.lightAmbientScale = scale
        
    def setLightDiffuseScale(self, scale):
        self.lightDiffuseScale = scale
        
    def setLightSpecularScale(self, scale):
        self.lightSpecularScale = scale
        
    def setLightSpecularPow(self, pow):
        self.lightSpecularPow = pow
        
    def setLightOcclusionScale(self, scale):
        self.lightOcclusionScale = scale   

    def setLightOcclusionRange(self, range):
        self.lightOcclusionRange = range   
        
    def setLightOcclusinResolution(self, resolution):
        self.lightOcclusinResolution = resolution   
        
    def setJointPrimitive(self, index, primitive):
        
        if index >=  self.jointPrimitives.shape[0]:
            return
        
        self.jointPrimitives[index] = primitive
        
    def setJointPrimitives(self, primitive):
        
        self.jointPrimitives[:] = primitive

    def setJointSize(self, index, size):
        
        if index >= self.jointPrimitives.shape[0]:
            return
        
        self.jointSizes[index] = size
        
    def setJointSizes(self, size):
        
        self.jointSizes[:] = size
        
    def setJointRounding(self, index, round):
        
        if index >= self.jointRoundings.shape[0]:
            return
        
        self.jointRoundings[index] = round
        
    def setJointRoundings(self, round):
        
        self.jointRoundings[:] = round
        
    def setJointSmoothing(self, index, smooth):
        
        if index >= self.jointRoundings.shape[0]:
            return
        
        self.jointSmoothings[index] = smooth
        
    def setJointSmoothings(self, smooth):
        
        self.jointSmoothings[:] = smooth   
        

    def setEdgePrimitive(self, index, primitive):
        
        if index >=  self.edgePrimitives.shape[0]:
            return
        
        self.edgePrimitives[index] = primitive
        
    def setEdgePrimitives(self, primitive):
        
        self.edgePrimitives[:] = primitive

    def setEdgeSize(self, index, size):
        
        if index >= self.edgeSizes.shape[0]:
            return
        
        self.edgeSizes[index] = size
        
    def setEdgeSizes(self, size):
        
        self.edgeSizes[:] = size
        
    def setEdgeRounding(self, index, round):
        
        if index >= self.edgeRoundings.shape[0]:
            return
        
        self.edgeRoundings[index] = round
        
    def setEdgeRoundings(self, round):
        
        self.edgeRoundings[:] = round
        
    def setEdgeSmoothing(self, index, smooth):
        
        if index >= self.edgeSmoothings.shape[0]:
            return
        
        self.edgeSmoothings[index] = smooth
        
    def setEdgeSmoothings(self, smooth):
        
        self.edgeSmoothings[:] = smooth   
        
    def setJointEdgeSmoothing(self, smooth):
        self.jointEdgeSmoothing = smooth
        
    def setGroundPrimitive(self, primitive):
        self.groundPrimitive = primitive    

    def setGroundPosition(self, position):
        self.groundPosition = position    
        
        self.updateGroundTransform()
 
    def setGroundRotation(self, rotation):
        self.groundRotation = rotation    
        
        self.updateGroundTransform()       
 
    def updateGroundTransform(self):
        
        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))
        
        groundRotMat = t3d.quaternions.quat2mat(self.groundRotation)
        groundTransMat = t3d.affines.compose(self.groundPosition, defaultRotMat, defaultScale)
        groundRotMat = t3d.affines.compose(defaultPos, groundRotMat, defaultScale)

        self.groundTransform = np.transpose(np.matmul(groundRotMat, groundTransMat))

    def setGroundSize(self, size):
        
        self.groundSize = size
        
    def setGroundRounding(self, round):
        
        self.groundRounding = round
        
    def setGroundSmoothing(self, smooth):
        
        self.groundSmoothing = smooth

