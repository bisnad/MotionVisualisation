from DMXEnttecPro import Controller
import json

config = { "port": "COM10",
          "light_properties_file": "configs/light_properties_beamZ_Panther15.json" }

class DMXController:
    def __init__(self, config):
        
        port = config["port"]
        light_properties_file = config["light_properties_file"]
        
        self.dmx_controller = Controller(port, auto_submit=False)
        self.loadLightProperties(light_properties_file)
        
    def close(self):
        
        self.dmx_controller.close()

    def loadLightProperties(self, file_path):
        
        with open(file_path) as json_data:
            self.light_properties = json.load(json_data)
            
        self.pan_range = self.light_properties["pan_angle_range"]
        self.tilt_range = self.light_properties["tilt_angle_range"]
        self.channel_count = self.light_properties["dmx_channel_count"]
        self.pan_channel = self.light_properties["dmx_pan_channel"]
        self.tilt_channel = self.light_properties["dmx_tilt_channel"]
        self.intensity_channel = self.light_properties["dmx_intensity_channel"]
        self.shutter_channel = self.light_properties["dmx_shutter_channel"]
        self.color_white_channel = self.light_properties["dmx_color_white_channel"]
        self.color_red_channel = self.light_properties["dmx_color_red_channel"]
        self.color_green_channel = self.light_properties["dmx_color_green_channel"]
        self.color_blue_channel = self.light_properties["dmx_color_blue_channel"]
            
    def set_shutter(self, light_nr, value):
        
        if self.shutter_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.shutter_channel, int(value * 255))
        
    def set_shutters(self, values):
        
        if self.shutter_channel == -1:
            return
        
        light_count = values.shape[0]
        for light_nr in range(light_count):
            self.dmx_controller.set_channel(light_nr * self.channel_count + self.shutter_channel, int(values[light_nr] * 255))
            
    def set_white(self, light_nr, value):
        
        if self.color_white_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.white_channel, int(value * 255))
        
    def set_whites(self, values):
        
        if self.color_white_channel == -1:
            return
        
        light_count = values.shape[0]
        for light_nr in range(light_count):
            self.dmx_controller.set_channel(light_nr * self.channel_count + self.white_channel, int(values[light_nr] * 255))
        
    def set_intensity(self, light_nr, value):
        
        if self.intensity_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.intensity_channel, int(value * 255))
        
    def set_intensities(self, values):
        
        if self.intensity_channel == -1:
            return
        
        light_count = values.shape[0]
        for light_nr in range(light_count):
            self.dmx_controller.set_channel(light_nr * self.channel_count + self.intensity_channel, int(values[light_nr] * 255))

    def set_color_red(self, light_nr, value):
        
        if self.color_red_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.color_red_channel, int(value * 255))
        
    def set_color_green(self, light_nr, value):
        
        if self.color_green_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.color_green_channel, int(value * 255))
        

    def set_color_blue(self, light_nr, value):
        
        if self.color_blue_channel == -1:
            return
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.color_blue_channel, int(value * 255))
        
        
    def set_pan_angle(self, light_nr, value):
        
        if self.pan_channel == -1:
            return
        
        dmx_value = int((value + self.pan_range / 2) / self.pan_range * 255)
        dmx_value = max(min(255, dmx_value), 0)
        
        #print("pan dmx_value ", dmx_value)
        
        self.dmx_controller.set_channel(light_nr * self.channel_count + self.pan_channel, dmx_value, submit_after=False)

    def set_pan_angles(self, values):
        
        if self.pan_channel == -1:
            return
        
        light_count = values.shape[0]
        for light_nr in range(light_count):
            dmx_value = int((values[light_nr] + self.pan_range / 2) / self.pan_range * 255)
            dmx_value = max(min(255, dmx_value), 0)
            
            self.dmx_controller.set_channel(light_nr * self.channel_count + self.pan_channel, dmx_value, submit_after=False)

    def set_tilt_angle(self, light_nr, value):
        
        if self.tilt_channel == -1:
            return
        
        dmx_value = int((value + self.tilt_range / 2) / self.tilt_range * 255)
        dmx_value = max(min(255, dmx_value), 0)
        
        #print("tilt dmx_value ", dmx_value)

        self.dmx_controller.set_channel(light_nr * self.channel_count + self.tilt_channel, dmx_value, submit_after=False)

    def set_tilt_angles(self, values):
        
        if self.tilt_channel == -1:
            return
        
        light_count = values.shape[0]
        for light_nr in range(light_count):
            dmx_value = int((values[light_nr] + self.tilt_range / 2) / self.tilt_range * 255)
            dmx_value = max(min(255, dmx_value), 0)
    
            self.dmx_controller.set_channel(light_nr * self.channel_count + self.tilt_channel, dmx_value, submit_after=False)

    def send(self):
        self.dmx_controller.submit()
            