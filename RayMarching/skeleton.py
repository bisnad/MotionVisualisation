import numpy as np
import transforms3d as t3d

def slerp(q0, q1, t=0.5, unit=True):
    """
    tested
    :param q0: shape = (*, n)
    :param q1: shape = (*, n)
    :param t: shape = (*)
    :param unit: If q0 and q1 are unit vectors
    :return: res: shape = (*, n)
    """
    eps = 1e-8
    if not unit:
        q0_n = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1_n = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    else:
        q0_n = q0
        q1_n = q1
    omega = np.arccos((q0_n * q1_n).sum(axis=-1).clip(-1, 1))
    dom = np.sin(omega)

    flag = dom < eps

    res = np.empty_like(q0_n)
    t_t = np.expand_dims(t[flag], axis=-1)
    res[flag] = (1 - t_t) * q0_n[flag] + t_t * q1_n[flag]

    flag = ~ flag

    t_t = t[flag]
    d_t = dom[flag]
    va = np.sin((1 - t_t) * omega[flag]) / d_t
    vb = np.sin(t_t * omega[flag]) / d_t
    res[flag] = (np.expand_dims(va, axis=-1) * q0_n[flag] + np.expand_dims(vb, axis=-1) * q1_n[flag])
    return res

class Skeleton():
    
    def __init__(self, jointFilter, jointConnectivity):

        self.jointFilter = jointFilter
        self.jointConnectivity = jointConnectivity
        
        self.skelTransform = np.eye(4)
        self.skelInvTransform = np.eye(4)
        
        self.jointCount = len(self.jointFilter)
        self.jointPositions = np.random.rand(self.jointCount, 3)
        self.jointRotations = np.random.rand(self.jointCount, 4)
        self.jointTransforms = np.zeros((self.jointCount, 4, 4))
        
        self.edgeCount = 0
        for jointChildren in self.jointConnectivity:
            self.edgeCount += len(jointChildren)
            
        self.edgeTransforms = np.zeros((self.edgeCount, 4, 4))
        self.edgeLengths = np.ones(self.edgeCount)
        
        self.udateSmoothing = 0.0
        
        print("skel jointCount ", self.jointCount, " edgeCount ", self.edgeCount)
        
    def setUpdateSmoothing(self, updateSmoothing):
        self.udateSmoothing = updateSmoothing
        
    def setPosition(self, position):

        self.skelTransform  = t3d.affines.compose(position, np.eye(3), np.ones((3)))
        self.skelInvTransform = t3d.affines.compose(position * -1.0, np.eye(3), np.ones((3)))

    def setJointPositions(self, positions):
        
        positions = positions[self.jointFilter, :]
        
        if positions.shape != self.jointPositions.shape:
            return

        self.jointPositions = self.jointPositions * self.udateSmoothing + positions * (1.0 - self.udateSmoothing)
        
        self.updateJointTransforms()
        self.updateEdgeTransforms()
        
    def setJointRotations(self, rotations):
        
        rotations = rotations[self.jointFilter, :]
        
        # prerotations of joints to align joint shapes
        
        # 0  : Hips
        rotations[0,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[0,:])
        
        # 6  : LeftShoulder
        rotations[6,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[6,:])
        # 7  : LeftArm
        rotations[7,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[7,:])
        # 8  : LeftForeArm
        rotations[8,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[8,:])
        # 9  : LeftForeArmRoll
        rotations[9,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[9,:])
        # 10  : LeftHand
        rotations[10,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[10,:])
        # 11  : LeftInHandMiddle
        rotations[11,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[11,:])
        # 12  : LeftHandMiddle2
        rotations[12,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[12,:])
        
        # 13  : RightShoulder
        rotations[13,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[13,:])
        # 14  : RightArm
        rotations[14,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[14,:])
        # 15  : RightForeArm
        rotations[15,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[15,:])
        # 16  : RightForeArmRoll
        rotations[16,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[16,:])
        # 17  : RightHand
        rotations[17,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[17,:])
        # 18  : RightInHandMiddle
        rotations[18,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[18,:])
        # 19  : RightHandMiddle2
        rotations[19,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[19,:])
        
        # 22 : LeftFoot
        rotations[22,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[22,:])
        # 23 : LeftToeBase
        rotations[23,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[23,:])
        
        # 26 : RightFoot
        rotations[26,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[26,:])
        # 27 : RightToeBase
        rotations[27,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[27,:])

        if rotations.shape != self.jointRotations.shape:
            return

        # TODO: address problem with rotation smoothing causes quick oscillations of some joints
        self.jointRotations = slerp(self.jointRotations, rotations, np.ones(self.jointCount) * (1.0 - self.udateSmoothing))
        self.jointRotations = self.jointRotations / np.linalg.norm(self.jointRotations)

        #self.jointRotations = rotations
        
        self.updateJointTransforms()
        self.updateEdgeTransforms()
        
    def updateJointTransforms(self):
        
        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))

        for jI in range(self.jointCount):
            
            jointPosition = self.jointPositions[jI]
            jointRotation = self.jointRotations[jI] # / np.linalg.norm(self.jointRotations[jI])
            jointRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), jointRotation)
            
            jointRotMat = t3d.quaternions.quat2mat(jointRotation)
            jointRotMat = t3d.affines.compose(defaultPos, jointRotMat, defaultScale)
   
            jointTransMat = t3d.affines.compose(jointPosition, defaultRotMat, defaultScale)

            self.jointTransforms[jI] = np.transpose(np.matmul(jointRotMat, np.matmul(self.skelTransform, jointTransMat)))


    def updateEdgeTransforms(self):

        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))
        refDir = np.array([0.0, 0.0, 1.0])
        
        eI = 0

        for pjI in range(self.jointCount):
            
            parentJointPos = self.jointPositions[pjI]
            parentJointRot = self.jointRotations[pjI] / np.linalg.norm(self.jointRotations[pjI])

            children = self.jointConnectivity[pjI]
            
            for cjI in children:
                
                childJointPos = self.jointPositions[cjI]
                
                edgePos = (parentJointPos + childJointPos) / 2
                
                edgeVec = childJointPos - parentJointPos
                edgeLength = np.linalg.norm(edgeVec)
                
                #print("pjI ", pjI, " cjI ", cjI, " edgeVec ", edgeVec, " edgeLength ", edgeLength)
                
                edgeRotation = self.jointRotations[pjI] # / np.linalg.norm(self.jointRotations[pjI])
                
                if pjI == 0 and cjI == 1: # hip to spine edge
                    edgeRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, -np.pi / 2.0, axes='sxyz'), edgeRotation)
                    
                edgeRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), edgeRotation)
                
                edgeRotMat = t3d.quaternions.quat2mat(edgeRotation)
                edgeRotMat = t3d.affines.compose(defaultPos, edgeRotMat, defaultScale)

                edgeTransMat = t3d.affines.compose(edgePos, defaultRotMat, defaultScale)
                
                self.edgeLengths[eI] = edgeLength
                
                #self.edgeTransforms[eI] = np.transpose(np.matmul(edgeRotMat, edgeTransMat))
                self.edgeTransforms[eI] = np.transpose(np.matmul(edgeRotMat, np.matmul(self.skelTransform, edgeTransMat)))
                
                eI += 1

    def getJointCount(self):
        return self.jointCount
    
    def getEdgeCount(self):
        return self.edgeCount
    
    def getEdgeLengths(self):
        return self.edgeLengths
    
    def getJointPositions(self):
        return self.jointPositions
    
    def getJointRotations(self):
        return self.jointRotations
    
    def getJointTransforms(self):
        return self.jointTransforms
    
    def getEdgeTransforms(self):
        return self.edgeTransforms
    
    
