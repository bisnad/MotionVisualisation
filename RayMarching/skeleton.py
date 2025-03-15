import numpy as np
import transforms3d as t3d
from common.quaternion import slerp
from enum import Enum

import json

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

class Skeleton():
    
    def __init__(self, jointFilter, jointConnectivity, hipJoints, jointRotCorrections):

        self.jointFilter = jointFilter
        self.jointConnectivity = jointConnectivity
        self.hipJoints = hipJoints
        self.jointRotCorrections = jointRotCorrections
        
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
        
    def initConfig(self, configFileName):
        
        with open(configFileName) as f:
            joint_settings = json.load(f)
            
        jointFilter = joint_settings["jointFilter"]
        jointConnectivity = joint_settings["jointConnectivity"]
        
        self.initTopology(jointFilter, jointConnectivity)

    def initTopology(self, jointFilter, jointConnectivity):
        
        self.jointFilter = jointFilter
        self.jointConnectivity = jointConnectivity
        
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
        
        #print("setJointPositions positions s ", positions.shape, " self.jointPositions s ", self.jointPositions.shape)
        
        positions = self.setAvatarJointPositions(positions)
    
        self.jointPositions = self.jointPositions * self.udateSmoothing + positions * (1.0 - self.udateSmoothing)
        
        self.updateJointTransforms()
        self.updateEdgeTransforms()
        
    def setJointRotations(self, rotations):
        
        #print("setJointRotations rotations s ", rotations.shape, " self.jointRotations s ", self.jointRotations.shape)
            
        rotations = self.setAvatarJointRotations(rotations)
 
        # TODO: address problem where rotation and position interpolation doesn't match
        self.jointRotations = slerp_pose(self.jointRotations, rotations, np.ones(self.jointCount) * (1.0 - self.udateSmoothing))
        self.jointRotations = self.jointRotations / np.linalg.norm(self.jointRotations)

        #self.jointRotations = rotations
        
        self.updateJointTransforms()
        self.updateEdgeTransforms()
        
    def setAvatarJointPositions(self, positions):
        
        #print("setAvatarJointPositions")
        #print("positions s ", positions.shape)
        #print("jointFilter l ", len(self.jointFilter))
        
        positions = positions[self.jointFilter, :]
        
        return positions

    def setAvatarJointRotations(self, rotations):
        
        rotations = rotations[self.jointFilter, :]
        
        for jI in range(rotations.shape[0]):
            jrc = self.jointRotCorrections[jI]
            rotations[jI,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(jrc[0], jrc[1], jrc[2], axes='sxyz'), rotations[jI,:])
        
        """
        # 0  : Hips
        #rotations[0,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[0,:])
        # 1  : RightUpLeg
        #rotations[1,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[1,:])
        # 2  : RightLeg
        #rotations[2,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[2,:])
        # 3  : RightFoot
        #rotations[3,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[3,:])
        # 4  : RightToeBase
        #rotations[4,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[4,:])
        # 5  : LeftUpLeg
        #rotations[5,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[5,:])
        # 6  : LeftLeg
        #rotations[6,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[6,:])
        # 7  : LeftFoot
        #rotations[7,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[7,:])
        # 8  : LeftToeBase
        #rotations[8,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[8,:])
        # 9  : Spine
        #rotations[9,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[9,:])
        # 10 : Spine1
        #rotations[10,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[10,:])
        # 11 : Spine2
        #rotations[11,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[11,:])
        # 12 : Spine3
        #rotations[12,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[12,:])
        # 13 : LeftShoulder
        rotations[13,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[13,:])
        # 14 : LeftArm
        rotations[14,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[14,:])
        # 15 : LeftForeArm
        rotations[15,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[15,:])
        # 16 : LeftHand
        rotations[16,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[16,:])
        # 17 : RightShoulder
        rotations[17,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[17,:])
        # 18 : RightArm
        rotations[18,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[18,:])
        # 19 : RightForeArm
        rotations[19,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[19,:])
        # 20 : RightHand
        rotations[20,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes='sxyz'), rotations[20,:])  
        # 21 : Neck
        rotations[21,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[21,:])
        # 22 : Head
        #rotations[22,:] = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), rotations[22,:])
        """

        return rotations

    def updateJointTransforms(self):
        
        defaultScale = np.ones((3))
        defaultRot = np.array([1.0, 0.0, 0.0, 0.0])
        defaultPos = np.array([0.0, 0.0, 0.0])
        defaultRotMat = (t3d.quaternions.quat2mat(defaultRot))

        for jI in range(self.jointCount):
            
            jointPosition = self.jointPositions[jI]
            jointRotation = self.jointRotations[jI] # / np.linalg.norm(self.jointRotations[jI])
            
            jointRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), jointRotation)
            
            """
            elif self.skeletonMode == SkeletonMode.SnakeAvatar:
                jointRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0, axes='sxyz'), jointRotation)
            """
            
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
                
                if pjI == self.hipJoints[0] and cjI == self.hipJoints[1]: # hip to RightUpLeg
                    edgeRotation = t3d.quaternions.qmult(t3d.euler.euler2quat(0.0, 0.0, -np.pi / 2.0, axes='sxyz'), edgeRotation)
                if pjI == self.hipJoints[0] and cjI == self.hipJoints[2]: # hip to LeftUpLeg
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
    
    
