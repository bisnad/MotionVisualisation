import threading
import numpy as np
import json

from pythonosc import dispatcher
from pythonosc import osc_server

config = { "dmx_controller": None,
          "light_setup": None,
          "ip": "127.0.0.1",
          "port": 9004}

class OscController:
    
    def __init__(self, config):
    
        self.dmx_controller = config["dmx_controller"]
        self.light_setup = config["light_setup"]
    
        self.mocap_joint_selected = 0
        self.light_selected = 0
    
        self._setup_osc_receive(config)

    def _setup_osc_receive(self, config):
        
        self.osc_receive_ip = config["ip"]
        self.osc_receive_port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/0/joint/pos_world", self.setMocapJointPositionsOsc)
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
        
    def setMocapJointPositionsOsc(self, address, *args):
        
        positions = np.array(args, dtype=np.float32)
        positions = np.reshape(positions, (-1, 3))
        
        self.light_setup.update_pan_tilt(positions[self.mocap_joint_selected:self.mocap_joint_selected+1])
        
        #self.dmx_controller.set_pan_angle(self.light_selected, np.degrees(self.light_setup.light_pans[0]))
        #self.dmx_controller.set_tilt_angle(self.light_selected, np.degrees(self.light_setup.light_tilts[0]))
        
        self.dmx_controller.set_pan_angle(self.light_selected, self.light_setup.light_pans[0])
        self.dmx_controller.set_tilt_angle(self.light_selected, self.light_setup.light_tilts[0])
        
        self.dmx_controller.send()

    def setMocapJointSelectOsc(self, address, *args):
        
        self.mocap_joint_selected = args[0]
 
    def setLightSelectOsc(self, address, *args):
        
        self.light_selected = args[0]       
        
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