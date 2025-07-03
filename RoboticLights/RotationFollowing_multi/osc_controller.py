import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server

config = { "dmx_controller": None,
          "ip": "127.0.0.1",
          "port": 9004}

class OscController:
    
    def __init__(self, config):
        
        self.dmx_controller = config["dmx_controller"]

        self.mocap_joint_selected = [ 0 ]
        self.light_selected = [ 0 ]

        self._setup_osc_receive(config)
        
    def _setup_osc_receive(self, config):
        
        self.osc_receive_ip = config["ip"]
        self.osc_receive_port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/0/joint/rot_world", self.setMocapJointRotationsOsc)
        self.dispatcher.map("/mocap/joint/select", self.setMocapJointSelectOsc)
        self.dispatcher.map("/light/select", self.setLightSelectOsc)
        
        self.dispatcher.map("/light/shutter", self.setLightShutterOsc)
        self.dispatcher.map("/light/white", self.setLightWhiteOsc)
        self.dispatcher.map("/light/intensity", self.setLightIntensityOsc)
        self.dispatcher.map("/light/red", self.setLightColorRedOsc)
        self.dispatcher.map("/light/green", self.setLightColorGreenOsc)
        self.dispatcher.map("/light/blue", self.setLightColorBlueOsc)
        self.dispatcher.map("/dmx/close", self.setDmxCloseOsc)

    def _start_osc_server(self):
        
        self.server = osc_server.ThreadingOSCUDPServer((self.osc_receive_ip, self.osc_receive_port), self.dispatcher)
        self.server.serve_forever()
        
    def start_osc_server(self):
        
        self.th = threading.Thread(target=self._start_osc_server)
        self.th.start()
        
    def stop_osc_server(self):
        
        self.server.shutdown()
        self.server.server_close()
        
    def _quaternions_to_euler(quaternions):
        """
        Convert an array of quaternions [x, y, z, w] to Euler angles [roll, pitch, yaw] in radians.
        Assumes 'xyz' intrinsic rotation order.
        """
        x = quaternions[:, 0]
        y = quaternions[:, 1]
        z = quaternions[:, 2]
        w = quaternions[:, 3]
    
        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
    
        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # Clamp for numerical stability
        pitch = np.arcsin(t2)
    
        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
    
        return np.stack([roll, pitch, yaw], axis=1)
        
    def _update_pan_tilt(self, rotations):
        
        euler_angles = self._quaternions_to_euler(rotations)
        
        roll = euler_angles[:,0]
        pitch = euler_angles[:,1]
        yaw = euler_angles[:,2]
        
        for light_selected, yaw, pitch in zip(self.lights_selected, yaw, pitch):
            self.dmx_controller.set_pan_angle(light_selected, np.degrees(yaw))
            self.dmx_controller.set_tilt_angle(light_selected, np.degrees(pitch))
            
        self.dmx_controller.send() 

    def setMocapJointRotationsOsc(self, address, *args):
        
        rotations = np.array(args, dtype=np.float32)
        rotations = np.reshape(rotations, (-1, 4))

        self._update_pan_tilt(rotations[self.mocap_joint_selected])

    def setMocapJointSelectOsc(self, address, *args):
        
        self.mocap_joint_selected = args
 
    def setLightSelectOsc(self, address, *args):
        
        self.light_selected = args    
        
    def setLightShutterOsc(self, address, *args):
            
        light_nr = args[0]
        shutter = args[1]
        
        self.dmx_controller.set_shutter(light_nr, shutter)
        self.dmx_controller.send()
        
    def setLightWhiteOsc(self, address, *args):
            
        light_nr = args[0]
        white = args[1]
        
        self.dmx_controller.set_white(light_nr, white)
        self.dmx_controller.send()
        
    def setLightIntensityOsc(self, address, *args):
            
        light_nr = args[0]
        intensity = args[1]
        
        self.dmx_controller.set_intensity(light_nr, intensity)
        self.dmx_controller.send()
        
    def setLightColorRedOsc(self, address, *args):
            
        light_nr = args[0]
        red = args[1]
        
        self.dmx_controller.set_color_red(light_nr, red)
        self.dmx_controller.send()
        
    def setLightColorGreenOsc(self, address, *args):
            
        light_nr = args[0]
        green = args[1]
        
        self.dmx_controller.set_color_green(light_nr, green)
        self.dmx_controller.send()
        
    def setLightColorBlueOsc(self, address, *args):
            
        light_nr = args[0]
        blue = args[1]
        
        self.dmx_controller.set_color_blue(light_nr, blue)
        self.dmx_controller.send()
        
    def setDmxCloseOsc(self, address, *args):
        
        self.dmx_controller.close()
        