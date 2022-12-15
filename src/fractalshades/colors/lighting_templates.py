# -*- coding: utf-8 -*-
import numpy as np
from fractalshades.colors.layers import Blinn_lighting
from fractalshades.utils import Protected_mapping

# Only diffuse light
pure_diffuse = Blinn_lighting(0.4, np.array([1., 1., 1.]))
pure_diffuse.add_light_source(
        k_diffuse=1.8,
        k_specular=0.0,
        shininess=0.,
        polar_angle=50.,
        azimuth_angle=20.,
        color=np.array([1.0, 1.0, 0.95]))

# Blinn lighting, rough material
rough = Blinn_lighting(0.4, np.array([1., 1., 1.]))
rough.add_light_source(
        k_diffuse=1.8,
        k_specular=2.0,
        shininess=15.,
        polar_angle=50.,
        azimuth_angle=20.,
        color=np.array([1.0, 1.0, 0.95]))

# Blinn lighting, glossy material
glossy = Blinn_lighting(0.4, np.array([1., 1., 1.]))
glossy.add_light_source(
        k_diffuse=1.8,
        k_specular=15.0,
        shininess=500.,
        polar_angle=50.,
        azimuth_angle=20.,
        color=np.array([1.0, 1.0, 0.95]))

# Blinn lighting, glossy material, rgb lights
rgb = Blinn_lighting(0.4, np.array([1., 1., 1.]))
rgb.add_light_source(
        k_diffuse=1.8,
        k_specular=50.0,
        shininess=500.,
        polar_angle=45.,
        azimuth_angle=20.,
        color=np.array([1.0, 0.0, 0.0]))
rgb.add_light_source(
        k_diffuse=1.8,
        k_specular=50.0,
        shininess=500.,
        polar_angle=55.,
        azimuth_angle=20.,
        color=np.array([0.0, 1.0, 0.0]))
rgb.add_light_source(
        k_diffuse=1.8,
        k_specular=50.0,
        shininess=500.,
        polar_angle=50.,
        azimuth_angle=35.,
        color=np.array([0.0, 0.0, 0.95]))

# To import a cmap one shall do:
# import fractalshades.colors as fscolors
# fscolors.lighting_register.keys()

lighting_register =  {
    "pure_diffuse": pure_diffuse,
    "rough": rough,
    "glossy": glossy,
    "rgb": rgb,
}
lighting_register = Protected_mapping(lighting_register)