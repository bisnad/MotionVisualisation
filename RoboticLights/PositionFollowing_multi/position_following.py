"""
Robotic Lights point towards the position target points in space.
Light control Data is sent directly via DMX to the lights using the EnttecPro USB to DMX Interface and/or via OSC to another computer
"""

"""
Imports
"""

import dmx_controller as dmx
import osc_controller as osc
import light_setup as ls

"""
Setup DMX Controller
"""

dmx.config["port"] = "COM10"
dmx.config["light_properties_file"] = "configs/light_properties_beamZ_Panther15.json"

dmx_controller = dmx.DMXController(dmx.config)

"""
Setup Lights
"""

ls.config["dmx_controller"] = dmx_controller
ls.config["light_setup_file"] = "configs/light_setup_single.json"

light_setup = ls.LightSetup(ls.config)

"""
Setup OSC Controller
"""

osc.config["dmx_controller"] = dmx_controller
osc.config["light_setup"] = light_setup
osc.config["ip"] = "127.0.0.1"
osc.config["port"] = 9004

osc_controller = osc.OscController(osc.config)
                                   

osc_controller.start_osc_server()






"""
osc_controller.stop_osc_server()
dmx_controller.close()
"""

                                   



