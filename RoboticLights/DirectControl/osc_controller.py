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

        self._setup_osc_receive(config)
        
    def _setup_osc_receive(self, config):
        
        self.osc_receive_ip = config["ip"]
        self.osc_receive_port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/light/pan", self.setLightPanOsc)
        self.dispatcher.map("/light/tilt", self.setLightTiltOsc)
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
        
    def setLightPanOsc(self, address, *args):
            
        light_nr = args[0]
        pan = args[1]
        
        self.dmx_controller.set_pan_angle(light_nr, pan)
        self.dmx_controller.send()

    def setLightTiltOsc(self, address, *args):
            
        light_nr = args[0]
        tilt = args[1]
        
        self.dmx_controller.set_tilt_angle(light_nr, tilt)
        self.dmx_controller.send()
        
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