# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses

from inspector import Func_inspector, best_match

#@dataclasses.dataclass
#class custom_data1:
#    f1: float# = 3.14158
#    f2: str
#    f3: str = "3.14158"
#
#@dataclasses.dataclass
#class custom_data2:
#    f1: float# = 3.14158
#    f2: str
#    f3: str = "3.14158"
#
#@dataclasses.dataclass
#class custom_data3(custom_data1):
#    f1: float# = 3.14158
#    f2: str
#    f3: str = "3.14158"
#
#def test_func_atomic():
#    def f(x: int=1, y: str="aa", z:float=1.0):
#        pass
#    fi = Func_inspector(f)
#    for key, val in fi.func_dict.items():
#        print(key, " --> ", val)
#        
#def test_func_union():
#    def f(x: typing.Union[int, float]=1., y: str="aa", z:typing.Optional[float]=1.0):
#        pass
#    fi = Func_inspector(f)
#    for key, val in fi.func_dict.items():
#        print(key, " --> ", val)
#    
#def test_best_match():
#    print(best_match(1., [int, float]))
#    print(best_match(1., [int, int]))
#    print(best_match(1, [int, float]))
#    print(best_match(1, [str, float]))
#    print(best_match(1.5, [int, float]))
#    print(best_match("1.5", [int, int, str]))
#    # print(best_match("1", [int, int]))
#    print(best_match(1.5, [int, int, complex]))
#
#    dc1 = custom_data3(1., "zaza")
#    print("dc1", dc1)
#    print(best_match(dc1, [int, int, complex, custom_data1]))
#    print(best_match(dc1, [int, int, complex, custom_data3]))
##    print(best_match
##    print(best_match(1.5, [int, int, complex]))1.5, [int, int, complex]))

if __name__ == "__main__":
    test_best_match()
    test_func_atomic()
    test_func_union()