# -*- coding: utf-8 -*-
import os
import unittest
import shutil

import numpy as np

import fractalshades
import fractalshades.utils as fsutils
import fractalshades.colors as fscolors
import fractalshades.models as fsm
from fractalshades.postproc import (
    Postproc_batch,
    Raw_pp,
    Attr_normal_pp,
    Attr_pp,
    Fractal_array
)
import test_config
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Blinn_lighting
)


class Test_layers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fractalshades.settings.enable_multiprocessing = True

        cls.colormap = fscolors.Fractal_colormap(
            colors=[[0.05, 0.5, 0.05],
                    [0.5, 0.5, .05],
                    [0.9, .9, 0.05],
                    [0.5, 0.05, 0.5],
                    [0.0, 0.5, 0.5]],
            kinds=['Lch', 'Lch', 'Lch', 'Lch'],
            grad_npts=[20, 20, 20,  20, 20],
            grad_funcs=['x', 'x', 'x', 'x'],
            extent='mirror'
        )        
        cls.colormap2 = fscolors.Fractal_colormap(
            colors=[[0.05, 0.05, 0.05],
                    [0.83, 0.69, .22]],
            kinds=['Lch'],
            grad_npts=[20],
            grad_funcs=['x'],
            extent='mirror'
        )

        subset_dir = os.path.join(test_config.temporary_data_dir, "_subset_dir")
        fsutils.mkdir_p(subset_dir)
        cls.subset_dir = subset_dir
        cls.calc_name = "test"
        cls.interior_calc_name = "test_int"
        cls.dir_ref = os.path.join(test_config.ref_data_dir, "subset_REF")

        x = -0.5
        y = 0.
        dx = 2.0
        nx = 1800
        cls.f = f = fsm.Mandelbrot(subset_dir)
        
        f.clean_up(cls.calc_name)
        f.clean_up(cls.interior_calc_name)
        
        f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
               theta_deg=0., projection="cartesian")
        f.calc_std_div(
            calc_name=cls.calc_name,
            subset=None,
            max_iter=1000,
            M_divergence=100.,
            epsilon_stationnary= 0.001,
        )

        interior = Fractal_array(f, cls.calc_name, "stop_reason",
                                 func= "x!=1")
        f.newton_calc(
            calc_name=cls.interior_calc_name,
            subset=interior,
            known_orders=None,
            max_order=250,
            max_newton=20,
            eps_newton_cv=1.e-12,
        )


    @test_config.no_stdout
    def test_interior_basic(self):
        """ Testing basic `Color_layer` plots """
        pp0 = Postproc_batch(self.f, self.calc_name)
        pp0.add_postproc("div", Raw_pp("stop_reason", func="x == 1."))

        pp = Postproc_batch(self.f, self.interior_calc_name)
        pp.add_postproc("order", Raw_pp("order",
                        func=lambda x: (x + 13) * 117 * np.pi % 90))
        pp.add_postproc("newton_cv", Raw_pp("stop_reason", func=lambda x: x != 1))
        pp.add_postproc("attr_map", Attr_normal_pp())

        plotter = fractalshades.Fractal_plotter([pp0, pp])

        plotter.add_layer(Bool_layer("div", output=False))
        plotter.add_layer(
            Color_layer("order",
                func= None,
                colormap=self.colormap,
                probes_z=[0., 0.01],
#                probes_kind="relative",
                output=True 
            )
        )
        plotter.add_layer(Normal_map_layer("attr_map", max_slope=80, output=False))
        plotter.add_layer(Bool_layer("newton_cv", output=False))

        plotter["order"].set_mask(
                plotter["newton_cv"],
                mask_color=(0.5, 0.5, 0.5)
        )

        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.8,
            k_specular=20.,
            shininess=800.,
            polar_angle = 120.,
            azimuth_angle= 25.,
            color=np.array([1.0, 1.0, 1.0]))
        
        plotter["order"].shade(plotter["attr_map"], light)
        plotter.plot()
        self.layer = plotter["order"]
        self.check_current_layer()


    @test_config.no_stdout
    def test_interior_basic2(self):
        """ Testing basic `Color_layer` plots """
        pp0 = Postproc_batch(self.f, self.calc_name)
        pp0.add_postproc("div", Raw_pp("stop_reason", func="x == 1."))

        pp = Postproc_batch(self.f, self.interior_calc_name)
        
        pp.add_postproc("newton_cv", Raw_pp("stop_reason", func=lambda x: x != 1))
        pp.add_postproc("attr_map", Attr_normal_pp())
        pp.add_postproc("attr", Attr_pp())

        plotter = fractalshades.Fractal_plotter([pp0, pp])

        plotter.add_layer(Bool_layer("div", output=False))
        plotter.add_layer(
            Color_layer(
                "attr",
                func= None,
                colormap=self.colormap2,
                probes_z=[0., 1.0],
#                probes_kind="relative",
                output=True
            )
        )
        plotter.add_layer(Normal_map_layer("attr_map", max_slope=80, output=False))
        plotter.add_layer(Bool_layer("newton_cv", output=False))

        plotter["attr"].set_mask(
                plotter["newton_cv"],
                mask_color=(0.5, 0.5, 0.5)
        )

        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.8,
            k_specular=20.,
            shininess=800.,
#            angles=(120., 25.),
            polar_angle = 120.,
            azimuth_angle= 25.,
            color=np.array([1.0, 1.0, 1.0]))
        
        plotter["attr"].shade(plotter["attr_map"], light)
        plotter.plot()
        self.layer = plotter["attr"]
        self.check_current_layer()


    def check_current_layer(self, err_max=0.01):
        """
        """
        layer = self.layer
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        test_file_path = os.path.join(self.subset_dir, file_name + ".png")
        ref_file_path = os.path.join(self.dir_ref, file_name + ".REF.png")
        err = test_config.compare_png(ref_file_path, test_file_path)
        print("err", err)


if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_layers]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_layers("test_interior_basic"))
        runner.run(suite)
