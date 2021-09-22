# -*- coding: utf-8 -*-
import os
import unittest
import shutil

import numpy as np

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.colors as fscolors
#import fractalshades.postproc as fspp
import fractalshades.models as fsm
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Raw_pp,
    Fieldlines_pp
)
import test_config
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Blinn_lighting
)


class Test_layers(unittest.TestCase):
    
    def setUp(self):
        layer_dir = os.path.join(test_config.temporary_data_dir, "_DEV")
        fsutils.mkdir_p(layer_dir)
        self.layer_dir = layer_dir
    
    def test_color_basic(self):
        """
        Testing basic plots
        """
        fs.settings.enable_multiprocessing = True
        
        x =  "-1.74928893611435556407228"
        y = "0."
        dx = "5.e-20" 
        
        
        x = '-1.36768994867991128'
        y = '0.00949048853859240532'
        dx = '2.477633848347765e-8'
        precision = 30
        
        
        x = "-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332"
        y = "0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529"
        dx = "1.8e-157"
        precision = 200
        
        nx = 1800

        f = fsm.Perturbation_mandelbrot(self.layer_dir)
        f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0, 
               precision=precision,
               theta_deg=0., projection="cartesian", antialiasing=False)
        
        f.calc_std_div(
            complex_type=np.complex128,
            calc_name="test",
            subset=None,
            max_iter=60000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-4,
            pc_threshold=0.1,
            SA_params={"cutdeg": 8,
                       "SA_err": 1.e-6,
                       "cutdeg_glitch": 8,
                       "use_Taylor_shift": True},
            glitch_eps=1.e-6,
            interior_detect=False,
            glitch_max_attempt=20)
        
#        f.base_calc(
#            calc_name="test",
#            subset=None,
#            max_iter=1000,
#            M_divergence=1000.,
#            epsilon_stationnary= 0.001,
#            datatype=np.complex128)
        f.clean_up("test")
        f.run()

        colormap = fscolors.Fractal_colormap(
            colors=[[1.        , 0.82352941, 0.25882353],
                    [0.70980392, 0.15686275, 0.38823529],
                    [0.27058824, 0.38039216, 0.49803922],
                    [0.04705882, 0.76078431, 0.77254902]],
            kinds=['Lab', 'Lch', 'Lch'],
            grad_npts=[100, 100, 100,  32],
            grad_funcs=['x', 'x', 'x**4'],
            extent='mirror'
        )
        colormap2 = fscolors.Fractal_colormap(
            colors=[[ 0.3, 0.3, 0.3],
                    [ 0.9, 0.9, 0.9,],
                    [ 0.3, 0.3, 0.3]],
            kinds=['Lch', 'Lch'],
            grad_npts=[100, 100, 100],
            grad_funcs=['1-(1-x**1.5)', 'x**1.5'],
            extent='mirror'
        )

        pp = Postproc_batch(f, "test")
        pp.add_postproc("cont_iter", Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        pp.add_postproc("field_lines", Fieldlines_pp(n_iter=10, swirl=1.0, damping_ratio=0.01))
        plotter = fs.Fractal_plotter(pp)

        plotter.add_layer(Color_layer("cont_iter", func="np.log(x)", colormap=colormap, output=True))
        plotter.add_layer(Color_layer("field_lines", func="np.mod(x, 2.*np.pi)", colormap=colormap2, output=True))
        plotter.add_layer(Bool_layer("interior", output=True))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=45, output=True))
        
        plotter["cont_iter"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
        plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
        plotter["field_lines"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
        
        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=20.,
            shininess=400.,
            angles=(135., 40.),
            coords=None,
            color=np.array([0.2, 0.2, 0.9])
        )
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=20.,
            shininess=400.,
            angles=(-45., 40.),
            coords=None,
            color=np.array([0.05, 0.05, 1.0])
        )
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=20.,
            shininess=400.,
            angles=(60., 40.),
            coords=None,
            color=np.array([1.0, 1.0, 0.5])
        )
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=20.,
            shininess=400.,
            angles=(60., 20.),
            coords=None,
            color=np.array([1.0, 0.0, 0.0])
        )
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=20.,
            shininess=400.,
            angles=(-60., 20.),
            coords=None,
            color=np.array([0.0, 1.0, 0.0])
        )
        plotter["cont_iter"].shade(plotter["DEM_map"], light)
        plotter["field_lines"].shade(plotter["DEM_map"], light)

        plotter.plot()





    def tearDown(self):
        layer_dir = self.layer_dir
        try:
            shutil.rmtree(os.path.join(layer_dir, "multiproc_calc"))
        except FileNotFoundError:
            pass

        
        



if __name__ == "__main__":
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_layers]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_layers("test_color_basic"))
        runner.run(suite)
