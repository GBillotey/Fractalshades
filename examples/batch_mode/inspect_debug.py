# -*- coding: utf-8 -*-
import os
import pickle

def read_FP(data_dir, data_file):
    save_path = os.path.join(data_dir, data_file)
    with open(save_path, 'rb') as tmpfile:
        FP_params = pickle.load(tmpfile)
        Z_path = pickle.load(tmpfile)
    return FP_params, Z_path


def debug():
    directory = "/home/geoffroy/Pictures/Fractal-shades/examples/batch_mode"
    data_dir = os.path.join(directory, "10-double_embedded_julia/data")
    data_file = "mandelbrot_pt0.ref"
    FP_params, Z_path = read_FP(data_dir, data_file)
    print("FP_params:\n", FP_params)
    print("Z_path:\n", Z_path)

if __name__ == "__main__":
    debug()
