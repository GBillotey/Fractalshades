# -*- coding: utf-8 -*-
"""Catalogue of functions to be used for GUI exploration

The following are implemented:
- Deepzoom_holomorphic
"""
import inspect
import typing
import os
import textwrap

import numpy as np
import mpmath

import fractalshades as fs
import fractalshades.settings
import fractalshades.colors
import fractalshades.gui

from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    Fieldlines_pp,
    DEM_pp,
    DEM_normal_pp,
    Raw_pp,
    Attr_pp,
    Attr_normal_pp,
    Fractal_array,
)

from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Grey_layer,
    Disp_layer,
    Virtual_layer,
    Overlay_mode,
    Blinn_lighting
)
from fractalshades.utils import Code_writer


script_header = f"""# -*- coding: utf-8 -*-
\"""============================================================================
Auto-generated from fractalshades GUI, version {fs.__version__}.
Save to `<file>.py` and run as a python script:
    > python <file>.py
============================================================================\"""
import os
import typing

import numpy as np
import mpmath
from PyQt6 import QtGui

import fractalshades
import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.gui as fsgui
import fractalshades.colors as fscolors

from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Fieldlines_pp,
    DEM_pp,
    Raw_pp,
    Attr_pp,
    Attr_normal_pp,
    Fractal_array
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Grey_layer,
    Disp_layer,
    Virtual_layer,
    Blinn_lighting,
    Overlay_mode
)

def plot(plot_dir):"""

script_footer = """
if __name__ == "__main__":
    # Sets `plot_dir` to the local directory where is saved this script, and
    # run the script
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    plot(plot_dir)
"""

def script_title_sep(title, indent=0):
    """ Return a script comment line with the given title """
    sep_line = " " * 4 * indent + "#" + ">" * (78 - 4 * indent) + "\n"
    return (
        "\n\n"
        + sep_line
        + " " * 4 * indent + "# " + title + "\n"
        + sep_line
    )


def script(source, kwargs, funcname):
    """ Writes source code for this script """
    func_params_str = script_assignments(kwargs, indent=1)
    source = (" " * 4) + source.replace("\n", "\n" + " " * 4)
    call_str = ",\n        ".join(k + " = " + k for k in kwargs.keys())
    call_str = f"    {funcname}(" + call_str + "\n    )\n"

    script = (
        script_header
        + script_title_sep("Parameters", 1)
        + func_params_str
        + script_title_sep("Plotting function", 1)
        + source
        + script_title_sep("Plotting call", 1)
        + call_str
        + script_title_sep("Footer", 0)
        + script_footer
        + "\n"
    )

    return script

def script_repr(obj, indent=0):
    """ Simple alias for Code_writer.var_tocode :
        string source code for an object
    """
    shift = " " * (4 * indent)
    code = Code_writer.var_tocode(obj)
    return shift + code.replace("\n", "\n" + shift)

def script_assignments(kwargs, indent=0):
    """ The parameter assignement part of the overall script:
        param1 = val1
        param2 = val2
        ...
    """
    shift = " " * (4 * indent)
    ret = "\n".join(
            [(k + " = " + script_repr(v)) for (k, v) in kwargs.items()]
    )
    ret = shift + ret.replace("\n", "\n" + shift)
    return ret


def getsource(callable_):
    """ Return the source code for this callable
    """
    if hasattr(callable_, "getsource"):
        return callable_.getsource()
    else:
        # Default: use inspect - use dedent to get consistent indentation level
        return textwrap.dedent(inspect.getsource(callable_))


def signature(callable_):
    """ Return the signature for this callable
    """
    if hasattr(callable_, "signature"):
        return callable_.signature()
    else:
        return inspect.signature(callable_)


class GUItemplate:
    """ Base class for all classes implementing a GUI-template function """

    def __init__(self, fractal):
        self.tuned_defaults = {}
        self.tuned_annotations = {}
        self.tuned_annotations_str = {}
        self.partial_vals = {}
        self.set_default("fractal", fractal)

    def set_default(self, pname, value):
        """ Change the default for param pname to tuned_annotations"""
        self.tuned_defaults[pname] = value
        
    def set_annotation(self, pname, tuned_annotations, annot_str):
        """ Change the annotation for param pname to tuned_annotations

        Parameters:
        -----------
        pname: str
            The name of the parameter to be modified
        tuned_annotations: annotation
            The modified annotation for this parameter
            https://docs.python.org/3/glossary.html#term-annotation
        annot_str: str
            string for this annotation (used for source code generation)
            for instance if annotation is `float`, just use "float"
        """
        self.tuned_annotations[pname] = tuned_annotations
        self.tuned_annotations_str[pname] = annot_str

    def make_partial(self, pname, val):
        """ Remove the parameter from the signature (hence from the
        GUI-settable parameters) and impose its value to be val.

        Implementation note: Value of the partials to be considered by the
        caller (caller responsability) before actually calling the GUItemplate
        """
        self.partial_vals[pname] = val

    def signature(self):
        """
        Signature used as a base for the GUI-display
        
        Returns:
        --------
        sgn, taking into account
            - params with modified default value (through set_default)
            - params with modified annotation (through set_annotation)
            - params suppressed as a result of partial
        """
        base_sgn = inspect.signature(self.__call__)
        my_defaults = self.tuned_defaults
        my_annots = self.tuned_annotations
        to_del = self.partial_vals.keys()

        sgn_params = []

        for p_name, param in base_sgn.parameters.items():
            
            if p_name in to_del:
                # Dropping this parameter -> skip to next item
                continue

            if p_name in my_defaults.keys():
                new_default = my_defaults[p_name]
            else:
                new_default = param.default

            if p_name in my_annots.keys():
                new_annot = my_annots[p_name]
            else:
                new_annot = param.annotation

            new_param = param.replace(
                default=new_default,
                annotation=new_annot,
            )
            sgn_params += [new_param]

        sgn = inspect.Signature(
                parameters=sgn_params,
                return_annotation=base_sgn.return_annotation
        )
        return sgn

    def getsource(self):
        """ Returns the source code defining the function
        """
        base_source = textwrap.dedent(inspect.getsource(self.__call__))
        func_name, func_params, func_body = self.split_source(base_source)
        
        my_defaults = self.tuned_defaults
        my_annots = self.tuned_annotations_str
        my_vals = self.partial_vals 
        
        # func name: changed from __call__ to the name of the class (mimics
        # a python function)
        func_name = f"def {self.__class__.__name__}(" 

        # func parameters:
        str_params = ""
        for p_name, v in func_params.items():
            if p_name == "self":
                continue

            decl, val = v.split("=") # dec -> <pname: annotation> / val -> <val>

            if p_name in my_defaults.keys():
                val = " " + Code_writer.var_tocode(my_defaults[p_name]) + ","
            elif p_name in my_vals.keys():
                val = " " + Code_writer.var_tocode(my_vals[p_name]) + ","
            elif p_name in my_annots.keys():
                decl = "    \n" + p_name + ": " +  my_annots[p_name]
                val = val + ","
            else:
                val = val + ","
            v = decl + "=" + val
            str_params += v

        return func_name + str_params + func_body


    def split_source(self, source):
        """
        Parameters
        ----------
        source: string
            Function source code  to be parsed

        Returns
        -------
        (func_name, func_params, func_body)

        func_name: str
            The extracted string "def func_name"

        func_params: mapping
            param -> param_txt where param_txt is the text
            describing the parameter in the function source code ie:
            "param_x: type = xxx" (as delimited by quotes)

        func_body: str
            the func body including the closing parameters parenthesis
        """
        paren_stack = []  # stack for ()
        bracket_stack = []  # stack for []
        braces_stack = []  # stack for {}
        
        name_ic = 0 # index of the first opening (
        func_params = {}
        body_ic = 0 # index of the matching )

        param_beg = 0 # position of the , or the ( for the forst param
        iparam = 0 # next_expected parameter
        
        # Signature of the UNbounded __call
        base_sgn = inspect.signature(self.__class__.__call__).parameters
        param_names = list(base_sgn.keys())
        n_params = len(param_names)
        
        started = False
        
        for ic, c in enumerate(source):
            if not started:
                if c == "(":
                    started = True
                    paren_stack.append(ic)
                    name_ic = param_beg = ic
                continue
            # Here we are in the parameters
            if c in "([{}])":
                if c == ")":
                    paren_stack.pop()
                elif c == "]":
                    bracket_stack.pop()
                elif c == "]":
                    braces_stack.pop()
                elif c == "(":
                    paren_stack.append(ic)
                elif c == "[":
                    bracket_stack.append(ic)
                elif c == "{":
                    braces_stack.append(ic)
                if len(paren_stack) == 0:
                    body_ic = ic
                    pname = param_names[iparam]
                    func_params[pname] = source[param_beg + 1: ic]
                    break # finished reading the parameter block

            may_end_param = (
                len(paren_stack) == 1
                and len(bracket_stack) == 0
                and len(braces_stack) == 0
            )

            if c == "," and may_end_param:
                # This is the end of a param declaration
                pname = param_names[iparam]
                func_params[pname] = source[param_beg + 1: ic]
                iparam += 1
                if iparam == n_params:
                    # because might have a last comma...
                    body_ic = ic
                    break
                param_beg = ic
            
        func_name = source[:name_ic]
        func_body = source[body_ic:]
        
        # Drop everything from the body before the ")" - avoid a potential 
        # ",," if the original params source code ends with ","
        incipit = func_body.find(")")
        func_body = func_body[incipit:]

        return func_name, func_params, func_body


#==============================================================================
colormap_int = fs.colors.Fractal_colormap(
    colors=[
        [1.        , 1.        , 1.        ],
        [0.16862746, 0.16862746, 0.16862746],
        [0.        , 0.        , 0.        ]
    ],
    kinds=['Lch', 'Lch'],
    grad_npts=[8, 8],
    grad_funcs=['x', 'x'],
    extent='mirror'
)

class std_zooming(GUItemplate):
    
    def __init__(self, fractal):
        """
A generic zooming function to explore standard or perturbation Fractals, 
intended for use with a GUI `fs.gui.guimodel.Fractal_GUI`

Compatible with :

    - holomorphic or non-holomorphic (Burning-ship & al) variants
    - optional fieldlines
    - optional shading
    - optional interior plots based on cycle attractivity & order
    - optional deepzoom implementation

Parameters
----------
fractal: `fs.Fractal`
    The fractal object to explore

Notes
----- 

.. note::
    
    A typical use case is show below (see also the interactive examples from
    the :doc:`../../examples/index/` section):

    ::

        fractal = fsm.Perturbation_mandelbrot(plot_dir)
        zooming = fs.gui.guitemplates.std_zooming(fractal)
        gui = fs.gui.guimodel.Fractal_GUI(zooming)
        gui.show()

"""
        super().__init__(fractal)

        badges = (
            "holomorphic", "implements_dzndc", "implements_fieldlines",
            "implements_newton", "implements_Milnor",
            "implements_interior_detection", "implements_deepzoom"
        )
        for attr in badges:
            setattr(self, attr, getattr(fractal, attr, False))

        self.connect_image_params = {
            "image_param": "calc_name"
        }

        if self.implements_deepzoom:
            self.connect_mouse_params = {
                "x": "x", "y": "y", "dx": "dx",
                "xy_ratio": "xy_ratio", "dps": "dps"
            }
        else:
            self.connect_mouse_params = {
                "x": "x", "y": "y", "dx": "dx",
                "xy_ratio": "xy_ratio", "dps": None
            }

        if not(self.implements_newton):
            # Delete all parameters associated with Newton
            self.make_partial("compute_newton", False)
            self.make_partial("_3", None)
            self.make_partial("max_order", None)
            self.make_partial("max_newton", None)
            self.make_partial("eps_newton_cv", None)
            self.make_partial("_7", None)
            self.make_partial("int_layer", None)
            self.make_partial("colormap_int", None)
            self.make_partial("cmap_func_int", None)
            self.make_partial("zmin_int", None)
            self.make_partial("zmax_int", None)
            self.make_partial("lighting_int", None)

        if self.holomorphic:
            # Delete all parameters associated with skew transform
            self.make_partial("_1b", None)
            self.make_partial("has_skew", False)
            self.make_partial("skew_00", 1.)
            self.make_partial("skew_01", 0.)
            self.make_partial("skew_10", 0.)
            self.make_partial("skew_11", 1.)

        if self.implements_dzndc != "user":
            self.make_partial("calc_dzndc", False)

        if not(self.implements_fieldlines):
            # Delete all parameters associated with fieldlines
            self.make_partial("_6", None)
            self.make_partial("has_fieldlines", False)
            self.make_partial("fieldlines_func", None)
            self.make_partial("fieldlines_kind", None)
            self.make_partial("fieldlines_zmin", None)
            self.make_partial("fieldlines_zmax", None)
            self.make_partial("backshift", None)
            self.make_partial("n_iter", None)
            self.make_partial("swirl", None)
            self.make_partial("damping_ratio", None)
            self.make_partial("twin_intensity", None)

        if self.implements_Milnor:
            self.set_annotation(
                "shading_kind",
                typing.Literal["potential", "Milnor"],
                'typing.Literal["potential", "Milnor"]'
            )

        if self.implements_interior_detection == "no":
            self.make_partial("interior_detect", False)
            self.make_partial("epsilon_stationnary", None)
        elif self.implements_interior_detection == "always":
            self.make_partial("interior_detect", True)
        elif self.implements_interior_detection == "user":
            pass
        else:
            raise ValueError(
                    "Unexpected value for GUI interior_detection:"
                    f"{self.implements_interior_detection}"
            )

        if self.implements_deepzoom:
            self.set_annotation("x", mpmath.mpf, "mpmath.mpf")
            self.set_annotation("y", mpmath.mpf, "mpmath.mpf")
            self.set_annotation("dx", mpmath.mpf, "mpmath.mpf")
            self.set_default("x", "0.0")
            self.set_default("y", "0.0")
            self.set_default("dx", "10.0")
        else:
            self.make_partial("dps", None)


    def __call__(
        self,
        fractal: fs.Fractal=None,
        calc_name: str="std_zooming_calc",

        _1: fs.gui.collapsible_separator="Zoom parameters",
        x: float = 0.0,
        y: float = 0.0,
        dx: float = 10.0,
        dps: int = 16,
        xy_ratio: float = 1.0,
        theta_deg: float = 0.0,
        nx: int = 600,

        _1b: fs.gui.collapsible_separator = (
                "Skew parameters /!\ Re-run when modified!"
        ),
        has_skew: bool = False,
        skew_00: float = 1.,
        skew_01: float = 0.,
        skew_10: float = 0.,
        skew_11: float = 1.,

        _2: fs.gui.collapsible_separator="Calculation parameters",
        max_iter: int = 5000,
        M_divergence: float = 1000.,
        interior_detect: bool = True,
        epsilon_stationnary: float = 0.001,
        calc_dzndc: bool = True,

        _3: fs.gui.collapsible_separator = "Newton parameters",
        compute_newton: bool = True,
        max_order: int = 1500,
        max_newton: int = 20,
        eps_newton_cv: float =1.e-8,

        _4: fs.gui.collapsible_separator="Plotting parameters: base field",
        base_layer: typing.Literal[
                 "continuous_iter",
                 "distance_estimation"
        ]="continuous_iter",

        colormap: fs.colors.Fractal_colormap=(
                fs.colors.cmap_register["classic"]
        ),
        cmap_func: fs.numpy_utils.Numpy_expr = (
                fs.numpy_utils.Numpy_expr("x", "np.log(x)")
        ),
        zmin: float = 0.0,
        zmax: float = 5.0,
        zshift: float = -1.0,
        mask_color: fs.colors.Color=(0.1, 0.1, 0.1, 1.0),

        _7: fs.gui.collapsible_separator="Plotting parameters: Newton field",
        int_layer: typing.Literal[
            "attractivity", "order", "attr / order"
        ]="attractivity",
        colormap_int: fs.colors.Fractal_colormap = (
                colormap_int
        ),
        cmap_func_int: fs.numpy_utils.Numpy_expr = (
                fs.numpy_utils.Numpy_expr("x", "x")
        ),
        zmin_int: float = 0.0,
        zmax_int: float = 1.0,

        _5: fs.gui.collapsible_separator = "Plotting parameters: shading",
        has_shading: bool = True,
        shading_kind: typing.Literal["potential"] = "potential", 
        lighting: Blinn_lighting = (
                fs.colors.lighting_register["glossy"]
        ),
        lighting_int: Blinn_lighting = (
                fs.colors.lighting_register["glossy"]
        ),
        max_slope: float = 60.,

        _6: fs.gui.collapsible_separator = "Plotting parameters: field lines",
        has_fieldlines: bool = False,
        fieldlines_func: fs.numpy_utils.Numpy_expr = (
                fs.numpy_utils.Numpy_expr("x", "x")
        ),
        fieldlines_kind: typing.Literal["overlay", "twin"] = "overlay",
        fieldlines_zmin: float = -1.0,
        fieldlines_zmax: float = 1.0,
        backshift: int = 3, 
        n_iter: int = 4,
        swirl: float = 0.,
        damping_ratio: float = 0.8,
        twin_intensity: float = 0.1,

    
        _8: fs.gui.collapsible_separator="High-quality rendering options",
        final_render: bool=False,
        supersampling: fs.core.supersampling_type = "None",
        jitter: bool = False,
        recovery_mode: bool = False,
        
        _9: fs.gui.collapsible_separator="Extra outputs",
        output_masks: bool = False,
        output_normals: bool = False,
        output_heightmaps: bool = False,
        hmap_mask: float = 0.,
        int_hmap_mask: float = 0.,
    
        _10: fs.gui.collapsible_separator="General settings",
        log_verbosity: typing.Literal[fs.log.verbosity_enum
                                      ] = "debug @ console + log",
        enable_multithreading: bool = True,
        inspect_calc: bool = False,
        no_newton: bool = False,
        postproc_dtype: typing.Literal["float32", "float64"] = "float32"
    ):

        fs.settings.log_directory = os.path.join(fractal.directory, "log")
        fs.set_log_handlers(verbosity=log_verbosity)
        fs.settings.enable_multithreading = enable_multithreading
        fs.settings.inspect_calc = inspect_calc
        fs.settings.no_newton = no_newton
        fs.settings.postproc_dtype = postproc_dtype


        zoom_kwargs = {
            "x": x,
            "y": y,
            "dx": dx,
            "nx": nx,
            "xy_ratio": xy_ratio,
            "theta_deg": theta_deg,
            "projection": "cartesian",
            "has_skew": has_skew,
            "skew_00": skew_00,
            "skew_01": skew_01,
            "skew_10": skew_10,
            "skew_11": skew_11
        }
        if fractal.implements_deepzoom:
            zoom_kwargs["precision"] = dps
        fractal.zoom(**zoom_kwargs)


        calc_std_div_kw = {
            "calc_name": calc_name,
            "subset": None,
            "max_iter": max_iter,
            "M_divergence": M_divergence,
        }


        if fractal.implements_dzndc == "user":
            calc_std_div_kw["calc_dzndc"] = calc_dzndc

        if shading_kind == "Milnor":
            calc_std_div_kw["calc_d2zndc2"] = True

        if has_fieldlines:
            calc_orbit = (backshift > 0)
            calc_std_div_kw["calc_orbit"] = calc_orbit
            calc_std_div_kw["backshift"] = backshift

        if fractal.implements_interior_detection == "always":
            calc_std_div_kw["epsilon_stationnary"] = epsilon_stationnary
        elif fractal.implements_interior_detection == "user":
            calc_std_div_kw["interior_detect"] = interior_detect
            calc_std_div_kw["epsilon_stationnary"] = epsilon_stationnary

        fractal.calc_std_div(**calc_std_div_kw)


        # Run the calculation for the interior points - if wanted
        if compute_newton:
            interior = Fractal_array(
                    fractal, calc_name, "stop_reason", func= "x != 1"
            )
            fractal.newton_calc(
                calc_name="interior",
                subset=interior,
                known_orders=None,
                max_order=max_order,
                max_newton=max_newton,
                eps_newton_cv=eps_newton_cv,
            )


        pp = Postproc_batch(fractal, calc_name)
        
        if base_layer == "continuous_iter":
            pp.add_postproc(base_layer, Continuous_iter_pp())
            if output_heightmaps:
                pp.add_postproc("base_hmap", Continuous_iter_pp())
    
        elif base_layer == "distance_estimation":
            pp.add_postproc("continuous_iter", Continuous_iter_pp())
            pp.add_postproc(base_layer, DEM_pp())
            if output_heightmaps:
                pp.add_postproc("base_hmap", DEM_pp())
    
        if has_fieldlines:
            pp.add_postproc(
                "fieldlines",
                Fieldlines_pp(n_iter, swirl, damping_ratio)
            )
        else:
            fieldlines_kind = "None"

        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1"))

        if compute_newton:
            pp_int = Postproc_batch(fractal, "interior")
            if int_layer == "attractivity":
                pp_int.add_postproc(int_layer, Attr_pp())
                if output_heightmaps:
                    pp_int.add_postproc("interior_hmap", Attr_pp())
            elif int_layer == "order":
                pp_int.add_postproc(int_layer, Raw_pp("order"))
                if output_heightmaps:
                    pp_int.add_postproc("interior_hmap", Raw_pp("order"))
            elif int_layer == "attr / order":
                pp_int.add_postproc(int_layer, Attr_pp(scale_by_order=True))
                if output_heightmaps:
                    pp_int.add_postproc(
                        "interior_hmap", Attr_pp(scale_by_order=True)
                    )
                
            # Set of unknown points
            pp_int.add_postproc(
                "unknown", Raw_pp("stop_reason", func="x == 0")
            )
            pps = [pp, pp_int]
        else:
            pps = pp
    
        if has_shading:
            pp.add_postproc("DEM_map", DEM_normal_pp(kind=shading_kind))
            if compute_newton:
                pp_int.add_postproc("attr_map", Attr_normal_pp())

        plotter = fs.Fractal_plotter(
            pps,
            final_render=final_render,
            supersampling=supersampling,
            jitter=jitter,
            recovery_mode=recovery_mode
        )

        # The mask values & curves for heighmaps
        r1 =  min(hmap_mask, 0.)
        r2 =  max(hmap_mask, 1.)
        dr = r2 - r1
        hmap_curve = lambda x : (np.clip(x, 0., 1.) - r1) / dr 

        r1 =  min(int_hmap_mask, 0.)
        r2 =  max(int_hmap_mask, 1.)
        dr = r2 - r1
        int_hmap_curve = lambda x : (np.clip(x, 0., 1.) - r1) / dr 


        # The layers
        plotter.add_layer(Bool_layer("interior", output=output_masks))

        if compute_newton:
            plotter.add_layer(Bool_layer("unknown", output=output_masks))

        if fieldlines_kind == "twin":
            plotter.add_layer(Virtual_layer(
                    "fieldlines", func=fieldlines_func, output=False
            ))
        elif fieldlines_kind == "overlay":
            plotter.add_layer(Grey_layer(
                    "fieldlines", func=fieldlines_func,
                    probes_z=[fieldlines_zmin, fieldlines_zmax],
                    output=False
            ))
    
        if has_shading:
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=max_slope, output=output_normals
            ))
            plotter["DEM_map"].set_mask(plotter["interior"])
            if compute_newton:
                plotter.add_layer(Normal_map_layer(
                    "attr_map", max_slope=90, output=output_normals
                ))

        if base_layer != 'continuous_iter':
            plotter.add_layer(
                Virtual_layer("continuous_iter", func=None, output=False)
            )

        plotter.add_layer(Color_layer(
                base_layer,
                func=cmap_func,
                colormap=colormap,
                probes_z=[zmin + zshift, zmax + zshift],
                output=True)
        )
        if output_heightmaps:
            plotter.add_layer(Disp_layer(
                    "base_hmap",
                    func=cmap_func,
                    curve=hmap_curve,
                    probes_z=[zmin + zshift, zmax + zshift],
                    output=True
            ))


        if compute_newton:
            plotter.add_layer(Color_layer(
                int_layer,
                func=cmap_func_int,
                colormap=colormap_int,
                probes_z=[zmin_int, zmax_int],
                output=False))
            plotter[int_layer].set_mask(plotter["unknown"],
                                        mask_color=mask_color)
            if output_heightmaps:
                plotter.add_layer(Disp_layer(
                        "interior_hmap",
                        func=cmap_func,
                        curve=int_hmap_curve,
                        probes_z=[zmin_int, zmax_int],
                        output=True
                ))
                plotter["interior_hmap"].set_mask(
                    plotter["unknown"],
                    mask_color=(int_hmap_mask,)
                )

        if fieldlines_kind == "twin":
            plotter[base_layer].set_twin_field(
                    plotter["fieldlines"], twin_intensity
            )
        elif fieldlines_kind == "overlay":
            overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.0)
            plotter[base_layer].overlay(plotter["fieldlines"], overlay_mode)
    
        if has_shading:
            plotter[base_layer].shade(plotter["DEM_map"], lighting)
            if compute_newton:
                plotter[int_layer].shade(plotter["attr_map"], lighting_int)
                plotter["attr_map"].set_mask(plotter["unknown"],
                                             mask_color=(0., 0., 0., 0.))

        if compute_newton:
            # Overlay : alpha composite with "interior" layer ie, where it is not
            # masked, we take the value of the "attr" layer
            overlay_mode = Overlay_mode(
                    "alpha_composite",
                    alpha_mask=plotter["interior"],
                    inverse_mask=True
            )
            plotter[base_layer].overlay(plotter[int_layer], overlay_mode=overlay_mode)
        else:
            plotter[base_layer].set_mask(
                plotter["interior"], mask_color=mask_color
            )

        if output_heightmaps:
            plotter["base_hmap"].set_mask(
                plotter["interior"], mask_color=(hmap_mask,)
            )

        plotter.plot()

        # Renaming output to match expected from the Fractal GUI
        layer = plotter[base_layer]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)

