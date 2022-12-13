# -*- coding: utf-8 -*-
import numpy as np
import unittest
import os
import shutil
import PIL

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.models as fsmodels
import fractalshades.colors as fscolors

import test_config

from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Raw_pp,
    Fieldlines_pp
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting
)


class Test_Perturbation_mandelbrot(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        fs.settings.enable_multiprocessing = True
        perturb_dir = os.path.join(test_config.temporary_data_dir,
                                 "_perturb_dir")
        fsutils.mkdir_p(perturb_dir)
        cls.perturb_dir = perturb_dir

        dir_ref = os.path.join(test_config.ref_data_dir, "perturb_REF")
        fsutils.mkdir_p(dir_ref)
        cls.dir_ref = dir_ref

        purple = np.array([150, 7, 71]) / 255.
        gold = np.array([229, 182, 31]) / 255.
        colors = np.vstack((purple[np.newaxis, :], gold[np.newaxis, :]))
        cls.colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x**2", extent="mirror")
        
    def tearDown(self):
        # Called after each test - cleaning the data directory
        data_dir = os.path.join(
            self.perturb_dir, self.test_name, "data"
        )
        shutil.rmtree(data_dir)


    # @test_config.no_stdout
    def test_M2_E20(self):
        """
        Testing all datatype options, 5e-20 test case."""
        x, y = "-1.74928893611435556407228", "0."
        dx = "5.e-20"
        precision = 30
        nx = 600
        test_name = self.test_M2_E20.__name__

        #  ("Xrange", np.complex128) deprecated
        for complex_type in [np.complex128]:
            if type(complex_type) is tuple:
                _, base_complex_type = complex_type
                calc_name = "Xr_" + np.dtype(base_complex_type).name
            else:
                calc_name = np.dtype(complex_type).name 
            with self.subTest(complex_type=complex_type):
                
                layer_name = test_name + "_potential_" + calc_name
                
                m = self.calc(x, y, dx, precision, nx, test_name,
                         calc_name)
                pp = Postproc_batch(m, calc_name)
                pp.add_postproc(layer_name, Continuous_iter_pp())
                pp.add_postproc("interior", Raw_pp("stop_reason",
                                       func=lambda x: np.isin(x, [0, 2])))
                pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
                
                plotter = fs.Fractal_plotter(pp)   
                plotter.add_layer(Bool_layer("interior", output=False))
                plotter.add_layer(Normal_map_layer(
                        "DEM_map", max_slope=30., output=True
                ))
                plotter.add_layer(Color_layer(
                        layer_name,
                        func=lambda x: np.log(x),
                        colormap=self.colormap,
                        probes_z=[6.196169853210449, 6.280143737792969],
                        output=True))
                plotter[layer_name].set_mask(plotter["interior"],
                                              mask_color=(0., 0., 0.))

                light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
                light.add_light_source(
                    k_diffuse=1.05,
                    k_specular=.0,
                    shininess=350.,
                    polar_angle=50.,
                    azimuth_angle=50.,
                    color=np.array([1.0, 1.0, 0.9])
                )
                light.add_light_source(
                    k_diffuse=0.,
                    k_specular=1.5,
                    shininess=350.,
                    polar_angle=50.,
                    azimuth_angle=40.,
                    color=np.array([1.0, 1.0, 0.9]),
                    material_specular_color=np.array([1.0, 1.0, 1.0])
                )
                plotter[layer_name].shade(plotter["DEM_map"], light)
                plotter.plot()

                self.layer = plotter[layer_name]
                self.test_name = test_name
                self.check_current_layer(err_max=0.04)


    @test_config.no_stdout
    def test_M2_int_E11(self):
        """
        Testing min without interior detection, 5e-12 test case.
        """
        test_name = self.test_M2_int_E11.__name__

        x, y = "-1.74920463345912691e+00", "-2.8684660237361114e-04"
        dx = "5e-12"
        precision = 16
        nx = 600

        # DEBUG point :
        fs.settings.enable_multiprocessing = True
        fs.settings.inspect_calc = True

        for BLA_eps in [1.e-6, None]:
            with self.subTest(BLA_eps=BLA_eps):
                if BLA_eps is None:
                    calc_name = "noBLA"
                else:
                    calc_name = str(BLA_eps) + "BLA"

                layer_name = test_name + "_potential_" + calc_name

                m = self.calc(x, y, dx, precision, nx, test_name,
                         calc_name, BLA_eps=BLA_eps)
                pp = Postproc_batch(m, calc_name)
                pp.add_postproc(layer_name, Continuous_iter_pp())
                pp.add_postproc("interior", Raw_pp("stop_reason",
                                       func=lambda x: np.isin(x, [0, 2])))
                pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

                plotter = fs.Fractal_plotter(pp)   
                plotter.add_layer(Bool_layer("interior", output=False))
                plotter.add_layer(Normal_map_layer("DEM_map", max_slope=30, output=True))
                plotter.add_layer(Color_layer(
                        layer_name,
                        func=lambda x: np.log(x),
                        colormap=self.colormap,
                        probes_z=[6.2335, 6.7],
                        output=True))
                plotter[layer_name].set_mask(plotter["interior"],
                                              mask_color=(0., 0., 1.))

                light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
                light.add_light_source(
                    k_diffuse=1.05,
                    k_specular=.0,
                    shininess=350.,
                    polar_angle=50.,
                    azimuth_angle=50.,
                    color=np.array([1.0, 1.0, 0.9]))
                light.add_light_source(
                    k_diffuse=0.,
                    k_specular=1.5,
                    shininess=350.,
                    polar_angle=50.,
                    azimuth_angle=40.,
                    color=np.array([1.0, 1.0, 0.9]),
                    material_specular_color=np.array([1.0, 1.0, 1.0])
                    )
                plotter[layer_name].shade(plotter["DEM_map"], light)
                plotter.plot()

                self.layer = plotter[layer_name]
                self.test_name = test_name
                self.check_current_layer(0.15)



    # @test_config.no_stdout
    def test_M2_E213(self):
        """
        Testing field lines, deep zoom
        """
        test_name = self.test_M2_E213.__name__
        calc_name = "tlaloc"

        x = '0.34570389446541448668331312104615893653459691273655833496059563502546829996045147247271327143565705645386902625644641809744568137675331733415337226841397624696255364683299529573163425022564963200471996897623616505588403854225'
        y = '-0.64581404795927501901833891812862865979727931493780907403982864606936141065500016197859928099369556416206459253378081052920526540908174111802042820970270682119655888546865599029297211693766764012377502232681203463021263672473'
        dx = '3.226224123547768e-213'
        precision = 224
        max_iter = 350000
        nx = 600 
#        SA_params = {"cutdeg": 64,
#                     "err": 1.e-6}
        c0 = np.array([1, 122, 193]) / 255.
        c1 = np.array([0, 54, 109]) / 255.
        c2 = np.array([5, 48, 99]) / 255.
        c3 = np.array([23, 175, 180]) / 255.
        c4 = np.array([0, 158, 130]) / 255.
        c5 = np.array([122, 175, 147]) / 255.
        c6 = np.array([151, 194, 138]) / 255.
        c7 = np.array([241, 249, 225]) / 255.
        c8 = np.array([111, 119, 0]) / 255.
        c9 = np.array([74, 190, 141]) / 255.

        colors = np.vstack((c0[np.newaxis, :],
                            c1[np.newaxis, :],
                            c2[np.newaxis, :],
                            c3[np.newaxis, :],
                            c4[np.newaxis, :],
                            c5[np.newaxis, :],
                            c6[np.newaxis, :],
                            c7[np.newaxis, :],
                            c8[np.newaxis, :],
                            c9[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x", extent="mirror")


        layer_name = test_name + "_potential_" + calc_name
        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name,
                 calc_fast=False, xy_ratio=1., max_iter=max_iter)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [0, 2])))
        pp.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=4, swirl=0., endpoint_k=0.4))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))


        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Virtual_layer("fieldlines", func="x-2.", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=40, output=True))
        # min: 11.806656837463379
        # max: 11.809494972229004
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[11.806656837463379,
                          11.806656837463379 * 0.7 + 0.30 * 11.809494972229004 - 0.00015],
                # probes_kind="relative",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 0.))
        plotter[layer_name].set_twin_field(plotter["fieldlines"], 0.00004)
        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=0.5,
            k_specular=0.0,
            shininess=400.,
            polar_angle=135.,
            azimuth_angle=60.,
            color=np.array([1.0, 1.0, 1.0]),
            )
        light.add_light_source(
            k_diffuse=0.2,
            k_specular=5000.0,
            shininess=800.,
            polar_angle=135.,
            azimuth_angle=18.,
            color=np.array([1.0, 1.0, 0.9]),
            )

        plotter[layer_name].shade(plotter["DEM_map"], light)

        plotter.plot()
        
        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.1)


    @test_config.no_stdout
    def test_glitch_divref(self):
        """
        Testing based on fail2 test case
        """
        test_name = self.test_glitch_divref.__name__
        calc_name = "fail2"
        
        x = '-1.36768994867991128'
        y = '0.00949048853859240532'
        dx = '2.477633848347765e-8'
        precision = 18
        nx = 1000

        black = np.array([0, 0, 0]) / 255.
        citrus2 = np.array([103, 189, 0]) / 255.
        # citrus2 = np.array([64, 109, 0]) / 255.
        
        colors = np.vstack((citrus2[np.newaxis, :],
                            # citrus3[np.newaxis, :],
                            black[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x", extent="mirror")
        
        
        layer_name = test_name + "_potential_" + calc_name
        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [0, 2])))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        
        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=30, output=True))
        # As divref glitch correction implies some random,
        # we test with absolute z range
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[7.2278738, 7.8332758],
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 0.))

        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.3,
            k_specular=25.0,
            shininess=150.,
            polar_angle=30.,
            azimuth_angle=50.,
            # angles=(30., 50.),
            color=np.array([1.0, 1.0, 1.0]),
            )
        light.add_light_source(
            k_diffuse=0.0,
            k_specular=3.0,
            shininess=150.,
            polar_angle=30.,
            azimuth_angle=50.,
#            angles=(30., 50.),
            color=np.array([1.0, 1.0, 1.0]),
            material_specular_color=np.array([1.0, 1.0, 1.0])
            )
        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter.plot()

        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.15)

    @test_config.no_stdout
    def test_supersampling(self):
        """
        Testing basic supersampling rendering
        """
        test_name = self.test_supersampling.__name__
        calc_name = "2-2"

        x = '-1.36768994867991128'
        y = '0.00949048853859240532'
        dx = '2.477633848347765e-8'
        precision = 18
        nx = 600

        colormap = fscolors.cmap_register["classic"]
        
        
        layer_name = test_name + "_potential_" + calc_name
        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [0, 2])))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        
        plotter = fs.Fractal_plotter(
                pp, supersampling="3x3", final_render=True
        )
        plotter.add_layer(Bool_layer("interior", output=False))

        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[7.2278738, 7.8332758],
#                probes_kind="absolute",
                output=True))
        plotter[layer_name].set_mask(
                plotter["interior"], mask_color=(0., 0., 0.)
        )
        plotter.plot()

        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.05)


    @test_config.no_stdout
    def test_glitch_divref5(self):
        """
        Testing based on fail5 test case
        """
        test_name = self.test_glitch_divref5.__name__
        calc_name = "fail5"
        
        x = '-1.8583883137588246496170054595'
        y = '0.000062628913856889509589212598191'
        dx = '4.688599337510725e-19'
        precision = 18
        nx = 1000

        black = np.array([0, 0, 0]) / 255.
        citrus2 = np.array([103, 189, 0]) / 255.
        # citrus2 = np.array([64, 109, 0]) / 255.
        
        colors = np.vstack((citrus2[np.newaxis, :],
                            # citrus3[np.newaxis, :],
                            black[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x", extent="mirror")
        
        
        layer_name = test_name + "_potential_" + calc_name
        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [0, 2])))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
        
        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=45, output=True))
        # As divref glitch correction implies some random,
        # we test with absolute z range
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[7.92, 7.93],
                #probes_kind="absolute",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 0.))

        light = Blinn_lighting(0.1, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.3,
            k_specular=25.0,
            shininess=150.,
#            angles=(30., 50.),
            polar_angle=30.,
            azimuth_angle=50.,
#            coords=None,
            color=np.array([1.0, 1.0, 1.0]),
            )
        light.add_light_source(
            k_diffuse=0.0,
            k_specular=3.0,
            shininess=150.,
#            angles=(30., 50.),
            polar_angle=30.,
            azimuth_angle=50.,
#            coords=None,
            color=np.array([1.0, 1.0, 1.0]),
            material_specular_color=np.array([1.0, 1.0, 1.0])
            )
        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter.plot()

        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.15)


    @test_config.no_stdout
    def test_glitch_dyn(self):
        """
        Testing based on Dinkydau "Flake" test case
        """
        test_name = self.test_glitch_dyn.__name__
        calc_name = "flake"
        
        # DEBUG point :
        fs.settings.enable_multiprocessing = True
        fs.settings.inspect_calc = True

        x = "-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332"
        y = "0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529"
        dx = "1.8e-157"
        precision = 200
        nx = 1600
#        SA_params = {"cutdeg": 32,
#                     "err": 1.e-6}
        c0 = np.array([242, 248, 163]) / 255.
        c1 = np.array([160, 105, 87]) / 255.
        c2 = np.array([202, 128, 21]) / 255.
        c3 = np.array([124, 2, 83]) / 255.
        c4 = np.array([35, 55, 61]) / 255.
        c5 = np.array([33, 146, 146]) / 255.
        c6 = np.array([5, 81, 22]) / 255. # blue 
        c7 = np.array([180, 16, 27]) / 255.
        c8 = np.array([63, 39, 53]) / 255.
        c9 = np.array([242, 248, 163]) / 255.

        colors = np.vstack((c0[np.newaxis, :],
                            c1[np.newaxis, :],
                            c2[np.newaxis, :],
                            c3[np.newaxis, :],
                            c4[np.newaxis, :],
                            c5[np.newaxis, :],
                            c6[np.newaxis, :],
                            c7[np.newaxis, :],
                            c8[np.newaxis, :],
                            c9[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x", extent="repeat")


        layer_name = test_name + "_potential_" + calc_name
        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name, 
                 calc_fast=True, xy_ratio=16/9.)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [0, 2])))


        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        # min: 10.321908950805664
        # max: 10.424850463867188
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[10.321908950805664, 
                          10.347048282623291],
                # probes_kind="absolute",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 0.))

        plotter.plot()
        
        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer()

    def test_deep_interior_detect(self):
        """
        Testing the interior early detection algo with perturbation & BLA on
        """
        test_name = self.test_deep_interior_detect.__name__

        x = "-1.9072475829123560219806014933444037282313"
        y = '7.54605431570315771754553123674989535657416e-16'
        dx = '3.660343578223062e-32'
        precision = 43
        nx = 600

#        # DEBUG point :
#        fs.settings.enable_multiprocessing = True
#        fs.settings.inspect_calc = True
#
#        for BLA_params in [{"eps": 1.e-6}, None]:
#            with self.subTest(BLA_params=BLA_params):
#                if BLA_params is None:
        calc_name = "deep_interior"
        layer_name = test_name + "_potential_" + calc_name
        
        white = np.array([255, 255, 255]) / 255.
        grey = np.array([0, 0, 0]) / 255.
        colors = np.vstack((white[np.newaxis, :], grey[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x**2", extent="mirror")

        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name, interior_detect=True, max_iter=1000000)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [2])))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=30,
                                           output=False))
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[10, 13.815954208374023],
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 1.))

        light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.05,
            k_specular=.0,
            shininess=350.,
            polar_angle=50.,
            azimuth_angle=50.,
            color=np.array([1.0, 1.0, 0.9]))
        light.add_light_source(
            k_diffuse=0.,
            k_specular=1.5,
            shininess=350.,
            polar_angle=50.,
            azimuth_angle=40.,
            color=np.array([1.0, 1.0, 0.9]),
            material_specular_color=np.array([1.0, 1.0, 1.0])
            )
        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter.plot()

        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.1)

    def test_ultradeep_interior_detect(self):
        """
        Testing the interior early detection algo with perturbation & BLA on
        """
        test_name = self.test_ultradeep_interior_detect.__name__

        x = '-1.999966194450370304184346885063505796755312415407248515117619229448015842423426843813761297788689138122870464065609498643538105757447721664856724960928039200970806596464697897247034380275662515774719795646696735873798312804539648952681115225456179242935106837745884878805854902169393836872097394050590046057699087967010196239765406551942511353248935870676912381954206583589473663772650104637785419392949872755058895530738089740079985776336454731048155381424443368009147832298545439060874543314328347318610344753331544040936498231198149727109'
        y = '0.0000000000000000000000000000000003001382436790938324072497303977592498734683119077333527017425728012047497561482358118564729928841407551922418650497818162547584808406226419681319987510966551024915920858367060072851094384239104512934585936736294083320495556911255641034741860247735534144716575799510702390797404223208558592193988956228128312647425358021104123150623316403610610181133505002078608735365638116235012859863602970282687765675758728641685358330772131661136113532338021950485899263595349420350880557330865130440135716258565997502285180701052571648187007997490077027'
        dx = '5.06722630e-433'
        precision = 550
        nx = 600

        calc_name = "ultradeep_interior"
        layer_name = test_name + "_potential_" + calc_name
        
        white = np.array([255, 255, 255]) / 255.
        grey = np.array([0, 0, 0]) / 255.
        colors = np.vstack((white[np.newaxis, :], grey[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x**2", extent="mirror")

        m = self.calc(x, y, dx, precision, nx, test_name,
                 calc_name, interior_detect=True, max_iter=100000000)
        pp = Postproc_batch(m, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                               func=lambda x: np.isin(x, [2])))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=25,
                                           output=False))
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[13.19380, 18.420959],
                # probes_kind="absolute",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                      mask_color=(0., 0., 1.))

        light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=2.05,
            k_specular=.0,
            shininess=350.,
            polar_angle=50.,
            azimuth_angle=10.,
            color=np.array([1.0, 1.0, 0.9]))
        light.add_light_source(
            k_diffuse=0.,
            k_specular=8.5,
            shininess=350.,
            polar_angle=50.,
            azimuth_angle=10.,
            color=np.array([1.0, 1.0, 0.9]),
            material_specular_color=np.array([1.0, 1.0, 1.0])
            )
        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter.plot()

        self.layer = plotter[layer_name]
        self.test_name = test_name
        self.check_current_layer(0.01)


    def calc(self, x, y, dx, precision, nx, test_name, calc_name,
            interior_detect=False, xy_ratio=1.0,
            BLA_eps=1.e-6, calc_fast=False, max_iter=50000):

        test_dir = os.path.join(self.perturb_dir, test_name)
        mandelbrot = fsmodels.Perturbation_mandelbrot(test_dir)
        mandelbrot.zoom(
             precision=precision,
             x=x,
             y=y,
             dx=dx,
             nx=nx,
             xy_ratio=xy_ratio,
             theta_deg=0.,
             projection="cartesian"
        )
        mandelbrot.clean_up(calc_name)

        if calc_fast: 
            mandelbrot.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=1.e3,
                epsilon_stationnary=1.e-3,
                interior_detect=False,
                BLA_eps=BLA_eps,
                calc_dzndc=False
            )
        else:
            mandelbrot.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=1.e3,
                epsilon_stationnary=1.e-3,
                interior_detect=interior_detect,
                BLA_eps=BLA_eps,
                calc_orbit=True,
                backshift=2
            )


        # mandelbrot.run()
        return mandelbrot

    def check_current_layer(self, err_max=0.01):
        """ Compare with stored reference image
        """
        layer = self.layer
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        test_file_path = os.path.join(
                self.perturb_dir,
                self.test_name,
                file_name + ".png")
        ref_file_path = os.path.join(self.dir_ref, file_name + ".REF.png")
        err = test_config.compare_png(ref_file_path, test_file_path)
        print("err < err_max", err, err_max)
        self.assertTrue(err < err_max)

if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_Perturbation_mandelbrot]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_Perturbation_mandelbrot("test_M2_E20"))
        # suite.addTest(Test_Perturbation_mandelbrot("test_ultradeep_interior_detect"))
        # suite.addTest(Test_Perturbation_mandelbrot("test_supersampling"))
        # suite.addTest(Test_Perturbation_mandelbrot("test_M2_E20"))
        runner.run(suite)

