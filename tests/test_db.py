# -*- coding: utf-8 -*-
"""
This file contains the tests for saving a db and plotting from it
"""
import sys
import os
import unittest

import numpy as np

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.colors as fscolors
import fractalshades.models as fsm
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Raw_pp,
    Fieldlines_pp,
    Attr_normal_pp,
    Attr_pp,
    Fractal_array
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting,
    Overlay_mode
)
import fractalshades.db

import test_config

class Test_db(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # DEBUG logs:
        fs.settings.enable_multiprocessing = False
        fs.settings.inspect_calc = True
        fs.settings.log_directory = os.path.join(
            test_config.temporary_data_dir, "_db_dir", "log"
        )
        fs.set_log_handlers(verbosity=3)

        db_dir = os.path.join(test_config.temporary_data_dir, "_db_dir")
        fsutils.mkdir_p(db_dir)
        cls.db_dir = db_dir
        cls.calc_name = "test"
        cls.dir_ref = os.path.join(test_config.ref_data_dir, "layers_REF")

        x = -0.5
        y = 0.
        dx = 5.
        nx = 600
        cls.f = f = fsm.Mandelbrot(db_dir)
        f.clean_up(cls.calc_name)

        cls.zoom_kwargs = zoom_kwargs = {
            "x": x,
            "y": y,
            "dx": dx,
            "nx": nx,
            "xy_ratio": 1.0,
            "theta_deg": 0.,
            "projection": fs.projection.Cartesian()
        }

        f.zoom(**zoom_kwargs)
        cls.frame = fs.db.Frame(
            x=0.,
            y=0.,
            dx=1.0,
            nx=zoom_kwargs["nx"],
            xy_ratio=1.0#zoom_kwargs["xy_ratio"]
        )


        f.calc_std_div(
            calc_name=cls.calc_name,
            subset=None,
            max_iter=20000,
            M_divergence=100.,
            epsilon_stationnary= 0.001,
            calc_orbit=True,
            backshift=3
        )

        cls.colormap = fscolors.Fractal_colormap(
            colors=[[1.        , 0.82352941, 0.25882353],
                    [0.70980392, 0.15686275, 0.38823529],
                    [0.27058824, 0.38039216, 0.49803922],
                    [0.04705882, 0.76078431, 0.77254902]],
            kinds=['Lab', 'Lch', 'Lch'],
            grad_npts=[100, 100, 100,  32],
            grad_funcs=['x', 'x', 'x**4'],
            extent='mirror'
        )
        cls.colormap2 = fscolors.Fractal_colormap(
            colors=[[ 0.8, 0.8, 0.8],
                    [ 0.2, 0.2, 0.2,],
                    [ 0.8, 0.8, 0.8]],
            kinds=['Lch', 'Lch'],
            grad_npts=[100, 100, 100],
            grad_funcs=['1-(1-x**1.5)', 'x**1.5'],
            extent='mirror'
        )
        cls.colormap_int = fscolors.Fractal_colormap(
            colors=[[0.05, 0.05, 0.05],
                    [0.2, 0.2, .2]],
            kinds=['Lch'],
            grad_npts=[20],
            grad_funcs=['x'],
            extent='mirror'
        )


    @test_config.no_stdout
    @unittest.skipIf(sys.platform.startswith("win"), "Fails on windows")
    def test_db_color_basic(self):
        # Fails under windows with 
        # FAILED tests/test_db.py::Test_db::test_db_color_basic - OSError: [Errno 22] Invalid argument: 'D:\\a\\Fractalshades\\Fractalshades\\tests\\_temporary_data\\_db_dir\\layers.db'
        """ Testing basic `Color_layer` plots from a saved database 
        Note: matrix test with supersampling & modifier to account for diff
        code paths
        """
        def plotter_modifier(plotter, time):
            """ A modifier that does nothing. It might ba attached to a Frame
            object"""
            pass
        
        for (ss, mod) in (
                (None, None),
                (None, plotter_modifier),
                ('3x3', None),
                ('3x3', plotter_modifier),
        ):
            with self.subTest(supersampling=ss, plotting_modifier=mod):

                pp = Postproc_batch(self.f, self.calc_name)

                pp.add_postproc("cont_iter", Continuous_iter_pp())
                pp.add_postproc("interior",
                                Raw_pp("stop_reason", func="x != 1."))

                plotter = fs.Fractal_plotter(
                    pp, final_render=True,
                    supersampling=ss,
                    recovery_mode=False
                )

                plotter.add_layer(Bool_layer("interior", output=True))
                plotter.add_layer(Color_layer(
                        "cont_iter",
                        func="np.log(x)",
                        colormap=self.colormap,
                        probes_z=[1.0511069297790527, 2.2134017944335938],
                        output=True
                ))
                plotter["cont_iter"].set_mask(
                        plotter["interior"],
                        mask_color=(0., 0., 0.)
                )
                
                if mod is None:
                    # Saving the rgb image as *.postdb
                    db_path = plotter.save_db(postdb_layer="cont_iter")
                else:
                    db_path = plotter.save_db()

                db = fs.db.Db(db_path)
                if mod is not None:
                    db.set_plotter(plotter, "cont_iter")
                img = db.plot(self.frame)
                
                mod_str = "frozen" if mod is None else "modified"

                out_file = os.path.join(
                    self.db_dir,
                    f"test_db_color_basic_{mod_str}_{ss}.png")
                img.save(out_file)
                
                ref_file = os.path.join(
                    self.dir_ref, f"Color_layer_cont_iter.REF.png")
                self.check_image(ref_file, out_file)


    @test_config.no_stdout
    @unittest.skipIf(sys.platform.startswith("win"), "Fails on windows")
    def test_db_overlay1(self):
        """ Testing a complex multilayered plot (with alpha compositing) from a
        saved database 

        Note: supersampling works but need carefull use of masks for the base
        layer:
            - masking at database creation for correct down-sampling
            - then un-masking for alpha_compositing (otherwise the mask being
              applied in the last step, will just erase the
              alpha-compositing
        """
        def plotter_modifier(plotter, time):
            """ A modifier that does nothing """
            pass
        
        for (ss, mod) in (
                (None, None),
                (None, plotter_modifier),
                ('2x2', None),
                ('2x2', plotter_modifier),
        ):
            with self.subTest(supersampling=ss, plotting_modifier=mod):

                layer_name = "overlay1"
                interior_calc_name = self.calc_name + "_over-1"
                f = self.f 
                interior = Fractal_array(f, self.calc_name, "stop_reason",
                                         func="x != 1.")
                f.newton_calc(
                    calc_name=interior_calc_name,
                    subset=interior,
                    known_orders=None,
                    max_order=5000,
                    max_newton=20,
                    eps_newton_cv=1.e-8,
                )
        
                pp0 = Postproc_batch(f, self.calc_name)
                pp0.add_postproc(layer_name, Continuous_iter_pp())
                pp0.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
                pp0.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
                pp0.add_postproc("div", Raw_pp("stop_reason", func="x == 1."))
                pp0.add_postproc(
                        "fieldlines",
                        Fieldlines_pp(n_iter=6, swirl=0.2, endpoint_k=0.3)
                )
        
                pp = Postproc_batch(f, interior_calc_name)
                pp.add_postproc("attr_map", Attr_normal_pp())
                pp.add_postproc("attr", Attr_pp())
        

                plotter = fs.Fractal_plotter(
                    [pp0, pp], final_render=True, supersampling=ss,
                    recovery_mode=False
                )   

                plotter.add_layer(Bool_layer("div", output=False))
                plotter.add_layer(Bool_layer("interior", output=True))
                plotter.add_layer(Color_layer(
                        layer_name,
                        func="np.log(x)",
                        colormap=self.colormap,
                        probes_z=[1.0511069297790527, 3.3979762077331546], # 0..0.4
                        output=True))
                plotter.add_layer(Color_layer(
                        "attr",
                        func=None, #"np.log(x)",
                        colormap=self.colormap_int,
                        probes_z=[0., 1.],
                        output=True))
        
                plotter.add_layer(Virtual_layer("fieldlines", func="x * 0.8 ", output=False))
                plotter[layer_name].set_twin_field(plotter["fieldlines"], 0.65)#1925)
        
                plotter.add_layer(Normal_map_layer("attr_map", max_slope=45, output=True))
                plotter.add_layer(Normal_map_layer("DEM_map", max_slope=38, output=True))
        
                light = Blinn_lighting(0.15, np.array([1., 1., 1.]))
                light.add_light_source(
                    k_diffuse=0.8,
                    k_specular=40.,
                    shininess=400.,
                    polar_angle=-40.,
                    azimuth_angle=25.,
                    color=np.array([1.0, 1.0, 0.8]))
                light.add_light_source(
                    k_diffuse=0.1,
                    k_specular=4.,
                    shininess=400.,
                    polar_angle=110.,
                    azimuth_angle=25.,
                    color=np.array([1.0, 0.0, 0.0]))
                light.add_light_source(
                    k_diffuse=0.1,
                    k_specular=3.,
                    shininess=400.,
                    polar_angle=130.,
                    azimuth_angle=25.,
                    color=np.array([0.0, 1.0, 0.0]))
                light.add_light_source(
                    k_diffuse=0.1,
                    k_specular=40.,
                    shininess=400.,
                    polar_angle=150.,
                    azimuth_angle=25.,
                    color=np.array([0.0, 0.0, 1.0]))
                plotter["attr"].shade(plotter["attr_map"], light)
                plotter["attr"].set_mask(plotter["div"],
                       mask_color=(0.0, 0.0, 0.0, 0.0))
                
                # Overlay : alpha composite
                overlay_mode = Overlay_mode("alpha_composite")
                plotter[layer_name].shade(plotter["DEM_map"], light)

                plotter[layer_name].overlay(
                        plotter["attr"],
                        overlay_mode=overlay_mode
                )
                
                
                if mod is None:
                    # Saving the rgb image as *.postdb
                    db_path = plotter.save_db(postdb_layer=layer_name)
                else:
                    db_path = plotter.save_db()

                db = fs.db.Db(db_path)

                if mod is not None:
                    db.set_plotter(plotter, layer_name)
            
                img = db.plot(self.frame)
                
                mod_str = "frozen" if mod is None else "modified"
                
                out_file = os.path.join(
                    self.db_dir,
                    f"test_db_overlay1_{mod_str}_{ss}.png")
                img.save(out_file)

                ref_file = os.path.join(
                    self.dir_ref, f"Color_layer_overlay1.REF.png")
                self.check_image(ref_file, out_file)


    def check_image(self, ref_file_path, test_file_path, err_max=0.01):
        """ Compare with stored reference image
        """
        err = test_config.compare_png(ref_file_path, test_file_path)
        print("err < err_max", err, err_max)
        self.assertTrue(err < err_max)






if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = True
    if full_test:
        runner.run(test_config.suite([Test_db]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_db("test_db_color_basic"))
        suite.addTest(Test_db("test_db_overlay1"))
        runner.run(suite)
    print("ok")
