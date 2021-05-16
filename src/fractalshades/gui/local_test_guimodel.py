# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses

from guimodel import Model

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
    
    
    
    
    
    

if __name__ == "__main__":
    test_nested_keys()