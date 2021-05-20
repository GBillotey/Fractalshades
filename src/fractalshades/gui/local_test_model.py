# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses
import functools

from model import Model, Func_submodel

def test_nested_keys():
    nested = {"a": {"z_pt1": 1., "zpt2": {("k_boum", 1): "zz",
                                          ("k_boum", 2): "zzz"}}}
    m = Model(nested)
    ret = m[("a", "z_pt1")]
    print("ret", ret)
    ret = m[("a", "zpt2", ("k_boum", 2))]
    print("ret", ret)

    m[("a", "zpt2", ("k_boum", 2))] = "Et voilou"
    ret = m[("a", "zpt2", ("k_boum", 2))]
    print("ret", ret)
    
def test_func_model():

    def f_atomic(x: int=1, yyyy: float=10., y: str="aa", z:float=1.0):
        pass 

    def f_union(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc"):
        pass 

    def f_optional(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc",
                option: typing.Optional[float]=12.354, option2: typing.Optional[float]=None):
        pass 
    
    def f_listed(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc",
                option: typing.Optional[float]=12.354, option2: typing.Optional[float]=None,
                listed: typing.Literal["a", "b", "c", 1, None]=None):
        pass 
    
    func = f_listed
    model = Model()
    func_smodel = Func_submodel(model, tuple(["func"]), func)
    print(func_smodel.func_dict)
    
def test_func_model_dataclass():

    @dataclasses.dataclass
    class custom_data1:
        f1: float = 1.120
        f2: str = "abc"
        f3: str = "3.14158"

    func = functools.partial(custom_data1.__init__, None)
    model = Model()
    func_smodel = Func_submodel(model, tuple(["func"]), func)
    print(func_smodel._dict)
        
    
        
    

if __name__ == "__main__":
#    test_nested_keys()
#    test_func_model()
    test_func_model_dataclass()