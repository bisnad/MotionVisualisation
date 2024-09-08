import numpy as np
import transforms3d as t3d
from common.quaternion import slerp

def slerp_pose(q0, q1, t=0.5):
    
    joint_count = q0.shape[0]
    
    qm = np.zeros_like(q0)
    
    for ji in range(joint_count): 
    
        current_quat = q0[ji, :]
        target_quat = q1[ji, :]
        
        quat_mix = t[ji]
        mix_quat = slerp(current_quat, target_quat, quat_mix )
        qm[ji, :] = mix_quat
        
    return qm

# def slerp(q0, q1, t=0.5, unit=True):
#     """
#     tested
#     :param q0: shape = (*, n)
#     :param q1: shape = (*, n)
#     :param t: shape = (*)
#     :param unit: If q0 and q1 are unit vectors
#     :return: res: shape = (*, n)
#     """
#     eps = 1e-8
#     if not unit:
#         q0_n = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
#         q1_n = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
#     else:
#         q0_n = q0
#         q1_n = q1
#     omega = np.arccos((q0_n * q1_n).sum(axis=-1).clip(-1, 1))
#     dom = np.sin(omega)

#     flag = dom < eps

#     res = np.empty_like(q0_n)
#     t_t = np.expand_dims(t[flag], axis=-1)
#     res[flag] = (1 - t_t) * q0_n[flag] + t_t * q1_n[flag]

#     flag = ~ flag

#     t_t = t[flag]
#     d_t = dom[flag]
#     va = np.sin((1 - t_t) * omega[flag]) / d_t
#     vb = np.sin(t_t * omega[flag]) / d_t
#     res[flag] = (np.expand_dims(va, axis=-1) * q0_n[flag] + np.expand_dims(vb, axis=-1) * q1_n[flag])
#     return res

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
        rotations[0,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[0,:])
        # 11 : Spine
        rotations[11,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[11,:])
        # 12 : Spine1
        rotations[12,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[12,:])
        # 13 : Spine2
        rotations[13,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[13,:])
        # 14 : Spine3
        rotations[14,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[14,:])
        # 25 : Neck
        rotations[25,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[25,:])
        # 26 : Head
        rotations[26,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[26,:])
        # 27 : Head_Nub
        rotations[27,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[27,:])
                                         
        # 1  : RightUpLeg
        rotations[1,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[1,:])
        # 2  : RightLeg
        rotations[2,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[2,:])
        # 3  : RightFoot
        #rotations[3,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[3,:])
        # 4  : RightToeBase
        rotations[4,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[4,:])
        # 5  : RightToeBase_Nub
        rotations[5,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[5,:])
        
        # 6  : LeftUpLeg
        rotations[6,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[6,:])
        # 7  : LeftLeg
        rotations[7,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[7,:])
        # 8  : LeftFoot
        #rotations[8,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[8,:])
        # 9  : LeftToeBase
        rotations[9,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[9,:])
        # 10 : LeftToeBase_Nub
        rotations[10,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[10,:])
        
        # 15 : LeftShoulder
        rotations[15,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[15,:])
        # 16 : LeftArm
        rotations[16,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[16,:])
        # 17 : LeftForeArm
        rotations[17,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[17,:])
        # 18 : LeftHand
        rotations[18,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[18,:])
        # 19 : LeftHand_Nub
        rotations[19,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[19,:])
        
    
        # 20 : RightShoulder
        rotations[20,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[20,:])
        # 21 : RightArm
        rotations[21,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[21,:])
        # 22 : RightForeArm
        rotations[22,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[22,:])
        # 23 : RightHand
        rotations[23,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[23,:])
        # 24 : RightHand_Nub
        rotations[24,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[24,:])       
 
        if rotations.shape != self.jointRotations.shape:
            return

        # TODO: address problem where rotation and position interpolation doesn't match
        self.jointRotations = slerp_pose(self.jointRotations, rotations, np.ones(self.jointCount) * (1.0 - self.udateSmoothing))
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
                
                if pjI == 0 and cjI == 1: # hip to RightUpLeg
                    edgeRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, -np.pi / 2.0, axes='sxyz'), edgeRotation)
                if pjI == 0 and cjI == 6: # hip to LeftUpLeg
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
    
    
