import json
import numpy as np
from scipy.spatial.transform import Rotation
import utils

config = { "dmx_controller": None,
          "light_setup_file": "configs/light_setup_circle8.json"}

class LightSetup:
    
    def __init__(self, config):
        
        self.light_pans = None
        self.light_tilts = None
        
        self.dmx_controller = config["dmx_controller"]
        
        self._loadLightSetup(config["light_setup_file"])
    
    def _loadLightSetup(self, file_path):
            
        with open(file_path) as json_data:
            self.light_setup = json.load(json_data)
            
        self.pos_scale = np.array(self.light_setup["pos_scale"])
        self.light_pan_flip = self.light_setup["light_pan_flip"]
        self.light_tilt_flip = self.light_setup["light_tilt_flip"]
        self.light_positions = np.array(self.light_setup["light_positions"])
        self.light_orientations = np.array(self.light_setup["light_orientations"])
        self.flip_block_active = np.array(self.light_setup["flip_block_active"])
        
        self.pan_range = self.dmx_controller.pan_range
        self.tilt_range = self.dmx_controller.tilt_range
        
        self.light_count = self.light_orientations.shape[0]
        self.prev_quat_w = [1.0] * self.light_count

        self.light_count = self.light_positions.shape[0]
        self.light_pans = np.zeros((self.light_count), dtype=np.float32)
        self.light_tilts = np.zeros((self.light_count), dtype=np.float32)
        
    # version adapted from from Incubatio
    def update_pan_tilt(self, target_positions):
        
        #print("target_positions ", target_positions)
        
        # number of target positions can't be larger than number of lights
        pos_count = min(target_positions.shape[0], self.light_count)
        
        for pos_nr in range(pos_count):
            
            #print("pos_nr ", pos_nr)
            
            # mocap position
            target_pos = np.copy(target_positions[pos_nr])
            
            #print("target_pos ", target_pos)
            
            # light position
            light_pos = np.copy(self.light_positions[pos_nr])
            
            #print("light_pos ", light_pos)

            #target_pos[2] += flock_height_offset # ??
            
            # flip x and y pos for target and light if lights are above the targets
            if light_pos[2] > target_pos[2]:
                # flip horizontal
                target_pos[0] *= -1.0
                light_pos[0] *= -1.0
    
                # flip vertical
                #light_pos[2] = grid_height - light_pos[2] ??

            # direction from light to target
            target_dir = target_pos - self.light_positions[pos_nr]
            
            # normalize direction
            target_dir = target_dir / np.linalg.norm(target_dir)
            
            #print("target_dir ", target_dir)

            # calculate spherical coordinates in radians
            light_rot_radian = utils.cart2spherical(target_dir)
            
            #print("light_rot_radian ", light_rot_radian)

            # convert radians to degrees
            light_rot_degrees = np.degrees(light_rot_radian)
                
            # convert form anticlockwise to clockwise rotation
            light_rot_degrees[0] *= self.light_pan_flip
            light_rot_degrees[1] *= self.light_tilt_flip
                
            """ 
            avoid pan euler angle discontinuities by converting pan rotation to quaternion
            check if the W-component of the current quaternion has a flipped sign with regards to the previous quaternion
            of the sign is flipped, unflip it
            then convert quaternion back to euler angle
            """
                
            rot_euler1 = light_rot_degrees[0]
            rot_quat = Rotation.from_euler("x", rot_euler1, degrees=True).as_quat()
                
            if self.prev_quat_w[pos_nr] < 0.0 and rot_quat[0] > 0.0:
                rot_quat[0] *= -1.0
            elif self.prev_quat_w[pos_nr] > 0.0 and rot_quat[0] < 0.0:
                rot_quat[0] *= -1.0
                    
            self.prev_quat_w[pos_nr] = rot_quat[0]
                
            rot_euler2 = Rotation.from_quat(rot_quat).as_euler("xyz", degrees=True)  
                    
            if self.flip_block_active == True:
                light_rot_degrees[0] = rot_euler2[0]
                
            #print("light_rot_degrees ", light_rot_degrees)
                
            self.light_pans[pos_nr] = light_rot_degrees[0]
            self.light_tilts[pos_nr] = light_rot_degrees[1]
            
            #print("self.light_pans ", self.light_pans)
            #print("self.light_tilts ", self.light_tilts)
