import dmx_controller as dmx
import osc_controller as osc


"""
Setup DMX Controller
"""

dmx.config["port"] = "COM10"
dmx.config["light_properties_file"] = "configs/light_properties_beamZ_Panther15.json"

dmx_controller = dmx.DMXController(dmx.config)

"""
Setup OSC Controller
"""

osc.config["dmx_controller"] = dmx_controller
osc.config["ip"] = "127.0.0.1"
osc.config["port"] = 9004

osc_controller = osc.OscController(osc.config)


osc_controller.start_osc_server()


"""
osc_controller.stop_osc_server()
dmx_controller.close()
"""

