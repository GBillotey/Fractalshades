# -*- coding: utf-8 -*-
"""
This contains the tests for the various graphical exports option,
including complex layering etc.
"""
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
    Fieldlines_pp,
    Attr_normal_pp,
    Attr_pp,
    Fractal_array
)
import test_config
from fractalshades.colors.layers import (
    Color_layer,
    Grey_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting,
    Overlay_mode
)


class Test_layers(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # DEBUG point :
        fs.settings.enable_multiprocessing = True
        fs.settings.inspect_calc = True

        layer_dir = os.path.join(test_config.temporary_data_dir, "_layers_dir")
        fsutils.mkdir_p(layer_dir)
        cls.layer_dir = layer_dir
        cls.calc_name = "test"
        cls.dir_ref = os.path.join(test_config.ref_data_dir, "layers_REF")

        x = -0.5
        y = 0.
        dx = 5.
        nx = 600
        cls.f = f = fsm.Mandelbrot(layer_dir)
        f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
               theta_deg=0., projection="cartesian", antialiasing=False)
        f.base_calc(
            calc_name=cls.calc_name,
            subset=None,
            max_iter=1000,
            M_divergence=100.,
            epsilon_stationnary= 0.001,
            )
        f.clean_up(cls.calc_name)
        f.run()
        
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
    def test_color_basic(self):
        """ Testing basic `Color_layer` plots """
        pp = Postproc_batch(self.f, self.calc_name)
        pp.add_postproc("cont_iter", Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Color_layer(
                "cont_iter",
                func="np.log(x)",
                colormap=self.colormap,
                probes_z=[0., 0.2],
                probes_kind="relative",
                output=True
        ))
        plotter["cont_iter"].set_mask(
                plotter["interior"],
                mask_color=(0., 0., 0.)
        )
        plotter.plot()
        self.layer = plotter["cont_iter"]
        self.check_current_layer()


    @test_config.no_stdout
    def test_normal_basic(self):
        """ Testing basic `Normal_map_layer` plots """
        pp = Postproc_batch(self.f, self.calc_name)
        pp.add_postproc("cont_iter", Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer(
                "DEM_map",
                max_slope=60,
                output=True
        ))
        plotter["DEM_map"].set_mask(
                plotter["interior"],
                mask_color=(0., 0., 0.)
        )
        plotter.plot()
        self.layer = plotter["DEM_map"]
        self.check_current_layer()


    @test_config.no_stdout
    def test_bool_basic(self):
        """ Testing basic `Bool_layer` masked plots """
        for (i, int_color) in enumerate([(0.5, 0., 0.), (0., 0., 0., 0.)]):
            with self.subTest(int_color=int_color):
                layer_name = "DEM_map-" + str(i + 1)
                pp = Postproc_batch(self.f, self.calc_name)
                pp.add_postproc("cont_iter", Continuous_iter_pp())
                pp.add_postproc("interior",
                                Raw_pp("stop_reason", func="x != 1."))
                pp.add_postproc(layer_name, DEM_normal_pp(kind="potential"))
                plotter = fs.Fractal_plotter(pp)   
                plotter.add_layer(Bool_layer("interior", output=False))
                plotter.add_layer(Normal_map_layer(
                        layer_name,
                        max_slope=90,
                        output=True))
                plotter[layer_name].set_mask(plotter["interior"],
                                             mask_color=int_color )
                plotter.plot()
                self.layer = plotter[layer_name]
                self.check_current_layer()


    @test_config.no_stdout
    def test_grey_basic(self):
        """ Testing basic `Bool_layer` masked plots """

        layer_name = "field_lines_grey"
        pp = Postproc_batch(self.f, self.calc_name)
        pp.add_postproc("cont_iter", Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
        pp.add_postproc(layer_name,
                Fieldlines_pp(n_iter=5, swirl=0.0, damping_ratio=0.1))
        plotter = fs.Fractal_plotter(pp)

        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Grey_layer(
                layer_name,
                func=None, #lambda x : np.cos(x),
                curve=None,
                probes_z=[0., 1.],
                probes_kind="relative",
                output=True))
        plotter[layer_name].set_mask(
                plotter["interior"],
                mask_color=0.2
        )
        plotter.plot()
        self.layer = plotter[layer_name]
        self.check_current_layer()


    @test_config.no_stdout
    def test_light_source(self):
        """ Testing basic `Blinn_lighting` plots """
        layer_name = "cont_iter_lighted"
        pp = Postproc_batch(self.f, self.calc_name)
        
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=60, output=True))
        plotter.add_layer(Color_layer(
                layer_name,
                func="np.log(x)",
                colormap=self.colormap,
                probes_z=[0., 0.8],
                probes_kind="relative",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                     mask_color=(0., 0., 0.))
        plotter["DEM_map"].set_mask(plotter["interior"],
                                    mask_color=(0., 0., 0.))
        
        light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=0.2,
            k_specular=10.,
            shininess=400.,
            angles=(-135., 20.),
            coords=None,
            color=np.array([0.05, 0.05, 1.0]))
        light.add_light_source(
            k_diffuse=0.2,
            k_specular=10.,
            shininess=400.,
            angles=(135., 20.),
            coords=None,
            color=np.array([0.5, 0.5, .4]))
        light.add_light_source(
            k_diffuse=1.3,
            k_specular=0.,
            shininess=0.,
            angles=(90., 40.),
            coords=None,
            color=np.array([1.0, 1.0, 1.0]))
        plotter[layer_name].shade(plotter["DEM_map"], light)

        plotter.plot()
        self.layer = plotter[layer_name]
        self.check_current_layer()


    @test_config.no_stdout
    def test_twin(self):
        """ Testing `Color_layer` `twin_field` method """
        layer_name = "cont_iter_twinned"
        pp = Postproc_batch(self.f, self.calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
        pp.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=4, swirl=0., damping_ratio=0.1))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Color_layer(
                layer_name,
                func="np.log(x)",
                colormap=self.colormap,
                probes_z=[0.0, 0.4], #[0.065, 0.465],
                probes_kind="relative",
                output=True))
        plotter.add_layer(Virtual_layer("fieldlines", func="x-2.2", output=False))

        plotter[layer_name].set_mask(plotter["interior"],
                                     mask_color=(0., 0., 0.))
        plotter[layer_name].set_twin_field(plotter["fieldlines"], 0.1)

        plotter.plot()
        self.layer = plotter[layer_name]
        self.check_current_layer()


    @test_config.no_stdout
    def test_overlay1(self):
        # lets do a second calculation for instance the interior points
        layer_name = "overlay1"
        interior_calc_name = self.calc_name + "_over-1"
        f = self.f 
        interior = Fractal_array(f, self.calc_name, "stop_reason",
                                 func= lambda x: (x!=1))
        f.newton_calc(
            calc_name=interior_calc_name,
            subset=interior,
            known_orders=None,
            max_order=250,
            max_newton=20,
            eps_newton_cv=1.e-12,
        )
        f.clean_up(interior_calc_name)
        f.run()

        pp0 = Postproc_batch(f, self.calc_name)
        pp0.add_postproc(layer_name, Continuous_iter_pp())
        pp0.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        pp0.add_postproc("div", Raw_pp("stop_reason", func="x == 1."))
        pp0.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=4, swirl=0., damping_ratio=0.1))

        pp = Postproc_batch(f, interior_calc_name)
        pp.add_postproc("attr_map", Attr_normal_pp())
        pp.add_postproc("attr", Attr_pp())

        plotter = fs.Fractal_plotter([pp0, pp])

        plotter.add_layer(Bool_layer("div", output=False))
        plotter.add_layer(Color_layer(
                layer_name,
                func="np.log(x)",
                colormap=self.colormap,
                probes_z=[0., 0.4],
                probes_kind="relative",
                output=True))
        plotter.add_layer(Color_layer(
                "attr",
                func=None, #"np.log(x)",
                colormap=self.colormap_int,
                probes_z=[0., 1.],
                probes_kind="relative",
                output=True))
        
        plotter.add_layer(Virtual_layer("fieldlines", func="x-2.2", output=False))
        plotter[layer_name].set_twin_field(plotter["fieldlines"], 0.1)

        plotter.add_layer(Normal_map_layer("attr_map", max_slope=90, output=True))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=60, output=True))

        light = Blinn_lighting(0.15, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=40.,
            shininess=400.,
            angles=(-40., 25.),
            coords=None,
            color=np.array([1.0, 1.0, 0.8]))
        light.add_light_source(
            k_diffuse=0.1,
            k_specular=4.,
            shininess=400.,
            angles=(110., 25.),
            coords=None,
            color=np.array([1.0, 0.0, 0.0]))
        light.add_light_source(
            k_diffuse=0.1,
            k_specular=3.,
            shininess=400.,
            angles=(130., 25.),
            coords=None,
            color=np.array([0.0, 1.0, 0.0]))
        light.add_light_source(
            k_diffuse=0.1,
            k_specular=40.,
            shininess=400.,
            angles=(150., 25.),
            coords=None,
            color=np.array([0.0, 0.0, 1.0]))
        plotter["attr"].shade(plotter["attr_map"], light)
        plotter["attr"].set_mask(plotter["div"],
               mask_color=(0.0, 0.0, 0.0, 0.0))
        
        # Overlay : alpha composite
        overlay_mode = Overlay_mode("alpha_composite")
        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter[layer_name].overlay(plotter["attr"], overlay_mode=overlay_mode)

        plotter.plot()
        self.layer = plotter[layer_name]
        self.check_current_layer()

    @test_config.no_stdout
    def test_overlay2(self):
        for (i, options) in enumerate([
                {"pegtop": 1.},
                {"Lch": 1.},
                {},
                ]):
            with self.subTest(options=options):
                layer_name = "tint_or_shade" + str(i + 1)
                pp = Postproc_batch(self.f, self.calc_name)
                pp.add_postproc(layer_name, Continuous_iter_pp())
                pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
                pp.add_postproc("fieldlines",
                        Fieldlines_pp(n_iter=4, swirl=0., damping_ratio=0.1))
        
                plotter = fs.Fractal_plotter(pp)   
                plotter.add_layer(Bool_layer("interior", output=False))
                plotter.add_layer(Color_layer(
                        layer_name,
                        func="np.log(x)",
                        colormap=self.colormap,
                        probes_z=[0., 0.4],
                        probes_kind="relative",
                        output=True))
                plotter.add_layer(Grey_layer("fieldlines", func=None, output=True))
        
                plotter[layer_name].set_mask(plotter["interior"],
                                             mask_color=(0., 0., 0.))

                # Overlay : tint_or_shade
                overlay_mode = Overlay_mode("tint_or_shade", **options)
                plotter[layer_name].overlay(plotter["fieldlines"], overlay_mode)
        
                plotter.plot()
                self.layer = plotter[layer_name]
                self.check_current_layer()
        
    @test_config.no_stdout
    def test_curve(self):
        for (i, curve) in enumerate([
                lambda x: x, # , + 1.,
                lambda x: 0.5 + (x - 0.5) * 0.2,
                lambda x: 0.5 + 0.5 * np.sign(x - 0.5) * np.abs((x - 0.5) * 2.)**2,
                lambda x: 0.5 + 0.5 * np.sign(x - 0.5) * np.abs((x - 0.5) * 2.)**0.5,
                ]):
            with self.subTest(curve=curve):
                layer_name = "curve" + str(i + 1)
                pp = Postproc_batch(self.f, self.calc_name)
                pp.add_postproc(layer_name, Continuous_iter_pp())
                pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
                pp.add_postproc("fieldlines",
                        Fieldlines_pp(n_iter=5, swirl=0.25*0.7, damping_ratio=0.1))

                plotter = fs.Fractal_plotter(pp)   
                plotter.add_layer(Bool_layer("interior", output=False))
                plotter.add_layer(Color_layer(
                        layer_name,
                        func="np.log(x)",
                        colormap=self.colormap,
                        probes_z=[0.0, 0.4], #[0., 0.4],
                        probes_kind="relative",
                        output=True))
                plotter.add_layer(Grey_layer("fieldlines",
                                             func=None,
                                             curve=curve,
                                             output=True))

                plotter[layer_name].set_mask(plotter["interior"],
                                             mask_color=(0., 0., 0.))

                # Overlay : tint_or_shade
                overlay_mode = Overlay_mode("tint_or_shade", Lch=1.)
                plotter[layer_name].overlay(plotter["fieldlines"], overlay_mode)

                plotter.plot()
                self.layer = plotter[layer_name]
                self.check_current_layer(0.04)


    def check_current_layer(self, err_max=0.01):
        """ Compare with stored reference image
        """
        layer = self.layer
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        test_file_path = os.path.join(self.layer_dir, file_name + ".png")
        ref_file_path = os.path.join(self.dir_ref, file_name + ".REF.png")
        err = test_config.compare_png(ref_file_path, test_file_path)
        self.assertTrue(err < err_max)



if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = True
    if full_test:
        runner.run(test_config.suite([Test_layers]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_layers("test_twin"))
        runner.run(suite)
