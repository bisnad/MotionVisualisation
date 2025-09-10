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

        self.mocap_joints_selected = [ 0 ]
        self.lights_selected = [ 0 ]

        self._setup_osc_receive(config)
        
    def _setup_osc_receive(self, config):
        
        self.osc_receive_ip = config["ip"]
        self.osc_receive_port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/*/joint/rot_world", self.setMocapJointRotationsOsc)
        #self.dispatcher.map("/mocap/*/joint/rot_local", self.setMocapJointRotationsOsc)
        self.dispatcher.map("/mocap/joint/select", self.setMocapJointsSelectOsc)
        self.dispatcher.map("/light/select", self.setLightsSelectOsc)
        
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
        
    def _quaternions_to_euler(self, quaternions):
        """
        Convert an array of quaternions [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
        Assumes 'xyz' intrinsic rotation order.
        """
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        
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
        
        min_count = min(len(rotations), len(self.lights_selected))

        _rotations = rotations[:min_count]
        _lights_selected = self.lights_selected[:min_count]
        
        euler_angles = self._quaternions_to_euler(_rotations)
        
        roll = euler_angles[:,0]
        pitch = euler_angles[:,1]
        yaw = euler_angles[:,2]
        
        #print("rotations ", rotations, " euler_angles ", euler_angles)

        for light_selected, yaw, pitch in zip(_lights_selected, yaw, pitch):
            self.dmx_controller.set_pan_angle(light_selected, np.degrees(yaw))
            self.dmx_controller.set_tilt_angle(light_selected, np.degrees(pitch))
            
        self.dmx_controller.send() 

    def setMocapJointRotationsOsc(self, address, *args):
        
        rotations = np.array(args, dtype=np.float32)
        rotations = np.reshape(rotations, (-1, 4))

        self._update_pan_tilt(rotations[self.mocap_joints_selected])

    def setMocapJointsSelectOsc(self, address, *args):
        
        self.mocap_joints_selected = list(args)
 
    def setLightsSelectOsc(self, address, *args):

        self.lights_selected = list(args)    
        
    def setLightShutterOsc(self, address, *args):
        
        if len(args) == 1:
            shutter = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_shutter(light_nr, shutter)
        elif len(args) == 2:
            light_nr = args[0]
            shutter = args[1]
            self.dmx_controller.set_shutter(light_nr, shutter)
        elif len(args) == len(self.lights_selected):
            shutters = args
            for light_nr, shutter in zip(self.lights_selected, shutters):
                self.dmx_controller.set_shutter(light_nr, shutter)

        self.dmx_controller.send()
        
    def setLightWhiteOsc(self, address, *args):
        
        if len(args) == 1:
            white = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_white(light_nr, white)
        elif len(args) == 2:
            light_nr = args[0]
            white = args[1]
            self.dmx_controller.set_white(light_nr, white)
        elif len(args) == len(self.lights_selected):
            whites = args
            for light_nr, white in zip(self.lights_selected, whites):
                self.dmx_controller.set_white(light_nr, white)

        self.dmx_controller.send()
        
    def setLightIntensityOsc(self, address, *args):
        
        if len(args) == 1:
            intensity = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_intensity(light_nr, intensity)
        elif len(args) == 2:
            light_nr = args[0]
            intensity = args[1]
            self.dmx_controller.set_intensity(light_nr, intensity)
        elif len(args) == len(self.lights_selected):
            intensities = args
            for light_nr, intensity in zip(self.lights_selected, intensities):
                self.dmx_controller.set_intensity(light_nr, intensity)

        self.dmx_controller.send()
        
    def setLightColorRedOsc(self, address, *args):
        
        if len(args) == 1:
            red = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_color_red(light_nr, red)
        elif len(args) == 2:
            light_nr = args[0]
            red = args[1]
            self.dmx_controller.set_color_red(light_nr, red) 
        elif len(args) == len(self.lights_selected):
            reds = args
            for light_nr, red in zip(self.lights_selected, reds):
                self.dmx_controller.set_color_red(light_nr, red)

        self.dmx_controller.send()
        
    def setLightColorGreenOsc(self, address, *args):
        
        if len(args) == 1:
            green = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_color_green(light_nr, green)
        elif len(args) == 2:
            light_nr = args[0]
            green = args[1]
            self.dmx_controller.set_color_green(light_nr, green) 
        elif len(args) == len(self.lights_selected):
            greens = args
            for light_nr, green in zip(self.lights_selected, greens):
                self.dmx_controller.set_color_green(light_nr, green)

        self.dmx_controller.send()
        
    def setLightColorBlueOsc(self, address, *args):
        
        if len(args) == 1:
            blue = args[0]
            for light_nr in self.lights_selected:
                self.dmx_controller.set_color_blue(light_nr, blue)
        elif len(args) == 2:
            light_nr = args[0]
            blue = args[1]
            self.dmx_controller.set_color_blue(light_nr, blue) 
        elif len(args) == len(self.lights_selected):
            blues = args
            for light_nr, blue in zip(self.lights_selected, blues):
                self.dmx_controller.set_color_blue(light_nr, blue)
            
        self.dmx_controller.send()
        
    def setDmxCloseOsc(self, address, *args):
        
        self.dmx_controller.close()
        