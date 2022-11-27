# -*- coding: utf-8 -*-
"""Catalogue of functions to be used for GUI exploration

The following are implemented:
- Deepzoom_holomorphic
"""
import inspect
import typing
import os
import sys
#import ast
import textwrap
#import pprint
import numbers

import numpy as np
#from PyQt6 import QtGui

import fractalshades as fs
# import fractalshades.models as fsm
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
    Virtual_layer,
    Overlay_mode,
    Blinn_lighting
)

import mpmath


def script_repr(obj, indent=0):
    """ Simple alias for Code_writer.var_tocode :
        string source code for an object
    """
    return Code_writer.var_tocode(obj, indent)

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
    ret.replace("\n", "\n" + shift)
    return ret

    
def getsource(callable_):
    """ Return the source code for this callable
    """
    if hasattr(callable_, "getsource"):
        return callable_.getsource()
    else:
        # Default: use inspect
        return inspect.getsource(callable_)


def signature(callable_):
    """ Return the signature for this callable
    """
#    print("***********IN SIGNATURE", type(callable_), hasattr(callable_, "signature"))
    if hasattr(callable_, "signature"):
        print(list(callable_.signature().parameters.keys()))
        return callable_.signature()
    else:
        return inspect.signature(callable_)


class Code_writer:
    """
    A set of static methods allowing to write Python source code
    """

    @staticmethod
    def var_tocode(var, indent=0):
        """
        Returns a string of python source code to serialize the variable var.

        Parameters
        ----------
        var: object
            The parameter to reconstruct. Supported types:
                None
                Numbers (int, float, bool)
                Dict
                list
                Class
        indent: int
            The current indentation level
        """
        shift = " " * (4 * indent)
        
        if var is None:
            return "None"
        if isinstance(var, numbers.Number):
            return repr(var)
        if isinstance(var, str):
            return f'"{var}"'
        if isinstance(var, mpmath.mpf):
            return repr(var)
        if isinstance(var, dict):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(k, indent+1)}: "
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for (k, v) in var.items()
            ])
            ret = f"{{\n{shift_inc}{ret}{shift}}}" # {{ for \{ in f-string
            return ret
        if isinstance(var, list):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for v in var
            ])
            ret = f"[\n{shift_inc}{ret}{shift}]" # {{ for \{ in f-string
            return ret
        if isinstance(var, tuple):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for v in var
            ])
            ret = f"(\n{shift_inc}{ret}{shift})" # {{ for \{ in f-string
            return ret

        if inspect.isclass(var):
            ret = f"{Code_writer.fullname(var)}"
            return ret

        if hasattr(var, "script_repr"):
            # Complex object: check first if has a dedicated script_repr
            # implementation
            return var.script_repr(indent)

        if hasattr(var, "init_kwargs"):
            # Complex object: defauts implementation serialize by calling the 
            # __init__ method
            return Code_writer.instance_tocode(var, indent)

        else:
            raise NotImplementedError(var)


    @staticmethod
    def fullname(class_):
        """ returns the fullname of a class """
        module = class_.__module__
        if module == 'builtins':
            return class_.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + class_.__qualname__


    @staticmethod
    def instance_tocode(obj, indent=0):
        """ Unserialize by calling init method
        """
        shift = " " * (4 * indent)
        fullname = Code_writer.fullname(obj.__class__)
        kwargs = obj.init_kwargs
        kwargs_code = Code_writer.func_args(kwargs, indent + 1)
        str_call_init = f"{shift}{fullname}(\n{kwargs_code}{shift})"
        return str_call_init


    @staticmethod
    def write_assignment(varname, value, indent=0):
        """
        %varname = %value
        """
        shift = " " * (4 * indent)

        try:
            var_str = Code_writer.var_tocode(value, indent)
        except NotImplementedError: # rethrow with hopefully better descr
            raise NotImplementedError(varname, value)
        str_assignment = f"{shift}{varname} = {var_str}\n"
        return str_assignment



#    @staticmethod
#    def write_fractal(var, indent=0):
#        shift = "\n" + " " * (4 * indent)
#        fractal_class = write_fractal.__class__
#        init_dic = 
#        s = inspect.signature(fractal_class.__init__)
        

#    @staticmethod
#    def write_func(funcname, impl, indent=0):
#        """
#        def funcname():
#            do stuff
#        """
#        shift = "\n" + " " * (4 * indent)
#        ret = textwrap.dedent(inspect.getsource(impl))
#        
#        # changing the name
#        beg = ret.find("(")
#        ret = f"def {funcname}{ret[beg:]}\n"
#
#        # Apply the indentation
#        ret = shift.join(l for l in ret.splitlines())
#        return shift + ret


#    @staticmethod
#    def call_func(funcname, kwargs, indent=0):
#        """
#        funcname(
#           key1=value1,
#           key2=value2,
#           ...
#        ):
#        """
#        shift = " " * (4 * indent)
#        func_args = Code_writer.func_args(kwargs, indent + 1)
#        str_call_func = f"{shift}ret = {funcname}(\n{func_args}{shift})"
#        return str_call_func

#    @staticmethod
#    def call_func(funcname, func_args, indent=0):
#        """
#        funcname(func_args)
#        """
#        shift = " " * (4 * indent)
#        shift_arg = " " * (4 * (indent+1))
#        func_args = func_args.replace(", ", ",")
#        func_args = (
#            shift_arg
#            + (",\n" + shift_arg).join(func_args.split(","))
#            + "\n"
#        )
#        
#        str_call_func = f"{shift}ret = {funcname}(\n{func_args}{shift})"
#        return str_call_func

    @staticmethod
    def func_args(kwargs, indent=0):
        """
           key1=value1,
           key2=value2,
        """
        shift = " " * (4 * indent)
        try:
            ret = shift.join([
                f"{k}={Code_writer.var_tocode(v)},\n"
                for (k, v) in kwargs.items()
            ])
        except NotImplementedError:
            etype, evalue, etraceback = sys.exc_info()
            raise  NotImplementedError(f"{evalue}  raised from {kwargs}")

        return shift + ret



class GUItemplate:
    """ Base class for all classes implementing a GUI-template function """

    def __init__(self, fractal):
        self.tuned_defaults = {}
        self.tuned_annotations = {}
        self.partial_vals = {}
        self.set_default("fractal", fractal)

    def set_default(self, pname, value):
        """ Change the default for param pname to tuned_annotations"""
        self.tuned_defaults[pname] = value
        
    def set_annotation(self, pname, tuned_annotations):
        """ Change the annotation for param pname to tuned_annotations"""
        self.tuned_annotations[pname] = tuned_annotations

    def make_partial(self, pname, val):
        """ Remove the parameter from the signature (hence from the
        GUI-settable parameters) and impose its value to be val.
        """
        self.partial_vals[pname] = val

    def signature(self):
        """
        Returns:
        --------
        sgn, taking into account
            - params with modified default value (through set_default)
            - params with modified annotation (through set_annotation)
            - params suppressed as a result of partial
        """
        print("in GUItemplate signature")
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
        base_source = textwrap.dedent(inspect.getsource(self.__call__))
        func_name, func_params, func_body = self.split_source(base_source)
        
        my_defaults = self.tuned_defaults
        my_annots = self.tuned_annotations
        my_vals = self.partial_vals 
        
        # func name: changed from __call__ to the name of the class (mimics
        # a python function)
        func_name = f"def {self.__class__.__name__}:" 

        # func parameters:
        str_params = ""
        for p_name, v in func_params.items():
            if p_name == "self":
                continue

            decl, val = v.split("=") # dec -> <pname: annotation> / val -> <val>

            if p_name in my_defaults.keys():
                val = " " + Code_writer.var_tocode(my_defaults[p_name]) + ",\n"
            elif p_name in my_vals.keys():
                val = " " + Code_writer.var_tocode(my_vals[p_name]) + ",\n"
            elif p_name in my_annots.keys():
                decl = p_name + ": " +  Code_writer.var_tocode(
                        my_annots[p_name]
                )
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
        print("body_ic2", body_ic)
        func_body = source[body_ic:]
        
        return func_name, func_params, func_body
        



class deepzooming(GUItemplate):
    
    def __init__(self, fractal):
        """
        A generic deepzoming explorer for Fractals implementing perturbation
        theory. Compatible with holomorphic or non-holomorphic (Burning-ship)
        variants
        """
        super().__init__(fractal)
        self.holomorphic = fractal.holomorphic

        self.connect_image_params = {
            "image_param": "calc_name"
        }
        self.connect_mouse_params = {
            "x": "x", "y": "y", "dx": "dx",
            "xy_ratio": "xy_ratio", "dps": "dps"
        }


    def __call__(
        self,

        fractal: fs.Fractal=None,
        calc_name: str="deepzoom",
    
        _1: fs.gui.collapsible_separator="Zoom parameters",
        x: mpmath.mpf="-1.0",
        y: mpmath.mpf="0.0",
        dx: mpmath.mpf="5.0",
        xy_ratio: float=1.0,
        theta_deg: float=0.0,
        dps: int=16,
        nx: int=600,
    
        _2: fs.gui.collapsible_separator="Calculation parameters",
        max_iter: int=5000,
        M_divergence: float=1000.,
        interior_detect: bool=True,
        epsilon_stationnary: float=0.001,
    
        _3: fs.gui.collapsible_separator="Bilinear series parameters",
        use_BLA: bool=True,
        eps: float=1.e-6,
    
        _4: fs.gui.collapsible_separator="Plotting parameters: base field",
        base_layer: typing.Literal[
                 "continuous_iter",
                 "distance_estimation"
        ]="continuous_iter",
        interior_mask: typing.Literal[
                 "all",
                 "not_diverging",
                 "dzndz_detection",
        ]="all",
        colormap: fs.colors.Fractal_colormap=(
                fs.colors.cmap_register["classic"]
        ),
        invert_cmap: bool=False,
        zmin: float=0.0,
        zmax: float=5.0,
    
        _5: fs.gui.collapsible_separator="Plotting parameters: shading",
        has_shading: bool=False,
        lighting: Blinn_lighting=(
                fs.colors.lighting_register["glossy"]
        ),
        max_slope: float = 60.,
    
        _6: fs.gui.collapsible_separator="Plotting parameters: field lines",
        has_fieldlines: bool=False,
        field_kind: typing.Literal["overlay", "twin"]="overlay",
        n_iter: int = 3,
        swirl: float = 0.,
        damping_ratio: float = 0.8,
        twin_intensity: float = 0.1,

        _7: fs.gui.collapsible_separator="Interior points",
        interior_color: fs.colors.Color=(0.1, 0.1, 0.1, 1.0),
    
    
        _8: fs.gui.collapsible_separator="High-quality rendering options",
        final_render: bool=False,
        supersampling: fs.core.supersampling_type = "None",
        jitter: bool = False,
        reload: bool = False,
    
        _9: fs.gui.collapsible_separator="General settings",
        log_verbosity: typing.Literal[fs.log.verbosity_enum
                                      ]="debug @ console + log",
        enable_multithreading: bool = True,
        inspect_calc: bool = False,
    ):

        fs.settings.log_directory = os.path.join(fractal.directory, "log")
        fs.set_log_handlers(verbosity=log_verbosity)
        fs.settings.enable_multithreading = enable_multithreading
        fs.settings.inspect_calc = inspect_calc

        fractal.zoom(
            precision=dps,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=xy_ratio,
            theta_deg=theta_deg,
            projection="cartesian",
        )
    
        BLA_eps = eps if use_BLA else None
    
        fractal.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=M_divergence,
                epsilon_stationnary=epsilon_stationnary,
                BLA_eps=BLA_eps,
                interior_detect=interior_detect,
            )
    
    
        pp = Postproc_batch(fractal, calc_name)
        
        if base_layer == "continuous_iter":
            pp.add_postproc(base_layer, Continuous_iter_pp())
    
        elif base_layer == "distance_estimation":
            pp.add_postproc("continuous_iter", Continuous_iter_pp())
            pp.add_postproc(base_layer, DEM_pp())
    
        if has_fieldlines:
            pp.add_postproc(
                "fieldlines",
                Fieldlines_pp(n_iter, swirl, damping_ratio)
            )
        else:
            field_kind = "None"
    
        interior_func = {
            "all": lambda x: x != 1,
            "not_diverging": lambda x: x == 0,
            "dzndz_detection": lambda x: x == 2,
        }[interior_mask]
        pp.add_postproc("interior", Raw_pp("stop_reason", func=interior_func))
    
        if has_shading:
            pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
    
        plotter = fs.Fractal_plotter(
            pp,
            final_render=final_render,
            supersampling=supersampling,
            jitter=jitter,
            reload=reload
        )
    
        plotter.add_layer(Bool_layer("interior", output=False))
    
        if field_kind == "twin":
            plotter.add_layer(Virtual_layer(
                    "fieldlines", func=None, output=False
            ))
        elif field_kind == "overlay":
            plotter.add_layer(Grey_layer(
                    "fieldlines", func=None, output=False
            ))
    
        if has_shading:
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=max_slope, output=True
            ))
    
        if base_layer != 'continuous_iter':
            plotter.add_layer(
                Virtual_layer("continuous_iter", func=None, output=False)
            )
    
        sign = {False: 1., True: -1.}[invert_cmap]
        plotter.add_layer(Color_layer(
                base_layer,
                func=lambda x: sign * np.log(x),
                colormap=colormap,
                probes_z=[zmin, zmax],
                output=True))
        plotter[base_layer].set_mask(
            plotter["interior"], mask_color=interior_color
        )
    
        if field_kind == "twin":
            plotter[base_layer].set_twin_field(plotter["fieldlines"],
                   twin_intensity)
        elif field_kind == "overlay":
            overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.0)
            plotter[base_layer].overlay(plotter["fieldlines"], overlay_mode)
    
        if has_shading:
            plotter[base_layer].shade(plotter["DEM_map"], lighting)
    
        plotter.plot()
        
        # Renaming output to match expected from the Fractal GUI
        layer = plotter[base_layer]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)



#==============================================================================


class shallowzooming(GUItemplate):
    
    def __init__(self, fractal):
        """
        A generic zooming explorer for standard Fractals

        Compatible with :
        - holomorphic or non-holomorphic (Burning-ship) variants
        - fieldlines implemented or not implements
        - interior cycle attractivity implement or not
        """
        super().__init__(fractal)
        self.holomorphic = fractal.holomorphic
        self.implements_fieldlines = fractal.implements_fieldlines
        self.implements_newton = fractal.implements_newton

        self.connect_image_params = {
            "image_param": "calc_name"
        }
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


    def __call__(
        self,

        fractal: fs.Fractal=None,
        calc_name: str="deepzoom",
    
        _1: fs.gui.collapsible_separator="Zoom parameters",
        x: float=-1.0,
        y: float=0.0,
        dx: float=5.0,
        xy_ratio: float=1.0,
        theta_deg: float=0.0,
        nx: int=600,

        _2: fs.gui.collapsible_separator="Calculation parameters",
        max_iter: int = 5000,
        M_divergence: float = 1000.,
        interior_detect: bool = True,
        epsilon_stationnary: float = 0.001,

        _3: fs.gui.collapsible_separator = "Newton parameters",
        compute_newton: bool = True,
        max_order: int = 1500,
        max_newton: int = 20,
        eps_newton_cv: float =1.e-12,

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
        zmin: float=0.0,
        zmax: float=5.0,
        mask_color: fs.colors.Color=(0.1, 0.1, 0.1, 1.0),

        _7: fs.gui.collapsible_separator="Plotting parameters: Newton field",
        int_layer: typing.Literal[
                 "attractivity",
                 "order"
        ]="attractivity",
        colormap_int: fs.colors.Fractal_colormap=(
                fs.colors.cmap_register["stellar"]
        ),
        cmap_func_int: fs.numpy_utils.Numpy_expr = (
                fs.numpy_utils.Numpy_expr("x", "x")
        ),
        zmin_int: float=0.0,
        zmax_int: float=5.0,

        _5: fs.gui.collapsible_separator="Plotting parameters: shading",
        has_shading: bool=True,
        lighting: Blinn_lighting=(
                fs.colors.lighting_register["glossy"]
        ),
        lighting_int: Blinn_lighting=(
                fs.colors.lighting_register["glossy"]
        ),
        max_slope: float = 60.,

        _6: fs.gui.collapsible_separator="Plotting parameters: field lines",
        has_fieldlines: bool=False,
        fieldlines_kind: typing.Literal["overlay", "twin"]="overlay",
        fieldlines_zmin: float=0.0,
        fieldlines_zmax: float=1.0,
        n_iter: int = 3,
        swirl: float = 0.,
        damping_ratio: float = 0.8,
        twin_intensity: float = 0.1,

    
        _8: fs.gui.collapsible_separator="High-quality rendering options",
        final_render: bool=False,
        supersampling: fs.core.supersampling_type = "None",
        jitter: bool = False,
        reload: bool = False,
        
        _9: fs.gui.collapsible_separator="Extra outputs",
        output_masks: bool = False,
        output_normals: bool = False,
        output_heightmaps: bool = False,
    
        _10: fs.gui.collapsible_separator="General settings",
        log_verbosity: typing.Literal[fs.log.verbosity_enum
                                      ]="debug @ console + log",
        enable_multithreading: bool = True,
        inspect_calc: bool = False,
    ):

        fs.settings.log_directory = os.path.join(fractal.directory, "log")
        fs.set_log_handlers(verbosity=log_verbosity)
        fs.settings.enable_multithreading = enable_multithreading
        fs.settings.inspect_calc = inspect_calc

        fractal.zoom(
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=xy_ratio,
            theta_deg=theta_deg,
            projection="cartesian",
        )
    
        fractal.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=M_divergence,
                epsilon_stationnary=epsilon_stationnary,
            )
        
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
    
        elif base_layer == "distance_estimation":
            pp.add_postproc("continuous_iter", Continuous_iter_pp())
            pp.add_postproc(base_layer, DEM_pp())
    
        if has_fieldlines:
            # Linear interpolation zmin -> 0. zmax -> 1.
            diff = fieldlines_zmax - fieldlines_zmin
            fieldlines_func = lambda x: (
                 (x - fieldlines_zmin) / diff
            )
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
            elif int_layer == "order":
                pp_int.add_postproc(int_layer, Raw_pp("order"))
                
            # Set of unknown points
            pp_int.add_postproc(
                "unknown", Raw_pp("stop_reason", func="x == 0")
            )
            pps = [pp, pp_int]
        else:
            pps = pp
    
        if has_shading:
            pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
            if compute_newton:
                pp_int.add_postproc("attr_map", Attr_normal_pp())

        plotter = fs.Fractal_plotter(
            pps,
            final_render=final_render,
            supersampling=supersampling,
            jitter=jitter,
            reload=reload
        )
    
    
        # The layers
        plotter.add_layer(Bool_layer("interior", output=False))

        if compute_newton:
            plotter.add_layer(Bool_layer("unknown", output=True))
            # plotter.add_layer(Bool_layer("div", output=True))

        if fieldlines_kind == "twin":
            plotter.add_layer(Virtual_layer(
                    "fieldlines", func=fieldlines_func, output=False
            ))
        elif fieldlines_kind == "overlay":
            plotter.add_layer(Grey_layer(
                    "fieldlines", func=fieldlines_func, output=False
            ))
    
        if has_shading:
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=max_slope, output=output_normals
            ))
            plotter["DEM_map"].set_mask(plotter["interior"],
                # mask_color=(0., 0., 0., 0.)
            )
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
                probes_z=[zmin, zmax],
                output=True)
        )

        if compute_newton:
            plotter.add_layer(Color_layer(
                int_layer,
                func=cmap_func_int,
                colormap=colormap_int,
                probes_z=[zmin_int, zmax_int],
                output=False))
            plotter[int_layer].set_mask(plotter["unknown"],
                                        mask_color=mask_color)

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

        plotter.plot()
        
        # Renaming output to match expected from the Fractal GUI
        layer = plotter[base_layer]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)

