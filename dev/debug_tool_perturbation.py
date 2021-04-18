#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:30:58 2021

@author: geoffroy
"""
import os
import pickle
import fractalshades.numpy_utils.xrange  as fsx

def reload_ref_point(save_path):
    """
    Reload arrays from a data file
       - params = main parameters used for the calculation
       - codes = complex_codes, int_codes, termination_codes
       - arrays : [Z, U, stop_reason, stop_iter]
    """
#    save_path = os.path.join("/home/geoffroy/Pictures/math/perturb/dev_SA003",
#                             "dev_pt0.ref")
    with open(save_path, 'rb') as tmpfile:
        FP_params = pickle.load(tmpfile)
        Z_path = pickle.load(tmpfile)
#        if FP_params["Xrange_complex_type"]:
#            pass
#            Z_path = np.asarray(Z_path)
    return FP_params, Z_path

def reload_SA(directory, iref, file_prefix):
    """
    """
    save_path =  os.path.join(directory, "data", file_prefix +
                        "_pt{0:d}.sa".format(iref))
    with open(save_path, 'rb') as tmpfile:
        SA_params = pickle.load(tmpfile)
    return SA_params

def test1():
    save_path_c1 = os.path.join("/home/geoffroy/Pictures/math/perturb/dev_SAd0001",
                             "data/dev_pt0.ref")
    save_path_c2 = os.path.join("/home/geoffroy/Pictures/math/perturb/dev_SAd0001",
                             "data/dev_pt1.ref")
    
    
    FP_params1, Z_path1 = reload_ref_point(save_path_c1)
    FP_params2, Z_path2 = reload_ref_point(save_path_c2)
    print("FP_params1:\n", FP_params1)
    print("zn1:\n", Z_path1[:, 0])
    
    print("FP_params2:\n", FP_params2)
    print("zn2:\n", Z_path2[:, 0])
    
    print(FP_params1['ref_point'] - FP_params2['ref_point'])   # (-2.15692821563586e-20 - 3.34079427820058e-21j)
    
def test2():
    directory = "/home/geoffroy/Pictures/math/perturb/dev_SAf0002"
    iref = 1
    file_prefix = "dev"
    SA_params = reload_SA(directory, iref, file_prefix)
    P_zn = SA_params["P"][0]
    print("P_zn\n", P_zn)

    P_dzndc = SA_params["P"][2]
    print("P_dzndc\n", P_dzndc)
    
    
    P_dzndc_est = P_zn.deriv() * (1. / SA_params["kc"])
    #P_dzndc_est.coeffs[0] = 0.
    print("P_dzndc_est\n", P_dzndc_est)
    print("diff\n", P_dzndc_est - P_dzndc)
    
    
    save_path_c1 = os.path.join(directory, "data/dev_pt1.ref")
    FP_params1, Z_path1 = reload_ref_point(save_path_c1)
    
    print("FP_params1", FP_params1)
    n_iter = SA_params["n_iter"]
    print("n_iter", n_iter)
    print("Z_path1", Z_path1[n_iter, :])

def test3():
    directory = "/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades/examples/flake"
    save_path0 = os.path.join(directory, "data/dev_pt0.ref")
    save_path1 = os.path.join(directory, "data/dev_pt1.ref")
    FP_params0, Z_path0 = reload_ref_point(save_path0)
    FP_params1, Z_path1 = reload_ref_point(save_path1)
    
    dc_ref = fsx.mpc_to_Xrange(FP_params1["ref_point"] - FP_params0["ref_point"])
    print("dc_ref", dc_ref)
    
    
    
    
    iref = 0
    file_prefix = "dev"
    SA_params = reload_SA(directory, iref, file_prefix)
    P0_zn = SA_params["P"][0]
    kc0 = SA_params["kc"]
    n_iter0 = SA_params["n_iter"]
    # print("P_zn\n", P0_zn)
    print("kc0\n", kc0, n_iter0, SA_params["iref"])
    

    iref = 1
    file_prefix = "dev"
    SA_params = reload_SA(directory, iref, file_prefix)
    P1_zn = SA_params["P"][0]
    kc1 = SA_params["kc"]
    n_iter1 = SA_params["n_iter"]
    print("dc_ref %", dc_ref / kc1)
    # print("P_zn\n", P1_zn)
    print("kc1\n", kc1, n_iter1, SA_params["iref"])
    P1_zn_scaled = P1_zn.scale_shift(kc1 / kc0)
    P0_zn_shifted = P0_zn.taylor_shift(dc_ref)# / kc0)
    
    print("zn1:\n", np.abs(Z_path1[31490:31510, 0]))
    print("zn1:\n", np.abs(Z_path1[0:10, 0]))
    print("div_iter:\n", FP_params1["div_iter"])

#    print("P0_zn", P0_zn.coeffs)
#    print("P1_zn", P1_zn.coeffs)
   # print("P0_zn_shifted", P0_zn.coeffs, P0_zn.coeffs._mantissa.dtype)
#    print("r", P1_zn_scaled.coeffs / P0_zn_shifted.coeffs)
    

#    iref = 101
#    file_prefix = "dev"
#    SA_params = reload_SA(directory, iref, file_prefix)
#    P101_zn = SA_params["P"][0]
#    print("##P_zn\n", P101_zn)



if __name__ == "__main__":
    test3()
    # Ball method 1 found period: 59444
    # SA running 212200 err:   2.80790428e-18 <<  6.90088067e-14
    # SA stop 212287  1.05937839e-13  2.60353573e-09
#    save_path = os.path.join("/home/geoffroy/Pictures/math/perturb/dev_SAd0001",
#                             "dev_pt0.ref")

#    print("Z_path :", Z_path.shape, type(Z_path), "\n", Z_path)
#    print("zn :\n", Z_path[:, 0])
#    print("dzndz :\n", Z_path[:, 1])
#    print("dzndc :\n", Z_path[:, 2])
#    print("d2zndc2 :\n", Z_path[:, 3])
#    print("d2zndc2 :\n", Z_path[:55000, 3])