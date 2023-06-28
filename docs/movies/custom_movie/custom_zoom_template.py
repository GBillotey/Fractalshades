# -*- coding: utf-8 -*-
"""============================================================================
Auto-generated from fractalshades GUI, version 1.1.0.
Save to `<file>.py` and use its plotter in the movie making main script
    > from <file> import get_plotter, plot_kwargs
============================================================================"""

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
import fractalshades.projection

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

# Note: in batch mode, edit this line to change the base directory
plot_dir = os.path.splitext(os.path.realpath(__file__))[0]

# Note: in batch mode, edit this line to change the local projection
# you may also call `plot` with a modified `batch_params` parameters
# (the latter allows to call from another module)
projection = fs.projection.Cartesian()

batch_params = {
    "projection": projection
}


#------------------------------------------------------------------------------
# Parameters - user editable 
#------------------------------------------------------------------------------
plot_kwargs = {
    "fractal": fractalshades.models.burning_ship.Perturbation_burning_ship(
        directory=plot_dir,
        flavor="Burning ship",
    ),
    "calc_name": "std_zooming_calc",
    "_1": "Zoom parameters",
    "x": "-1.939974858704598622717653847018167050885779075",
    "y": "0.00119716141570012717802412988272794623070942673",
    "dx": "1.9962495749473e-36",
    "dps": 45,
    "xy_ratio": 1.0,
    "theta_deg": 0.0,
    "nx": 600,
    "_1b": "Skew parameters /!\ Re-run when modified!",
    "has_skew": True,
    "skew_00": -0.051385354178351855,
    "skew_01": 1.1933398744211905,
    "skew_10": -0.9017294557172145,
    "skew_11": 1.4803773694629254,
    "_2": "Calculation parameters",
    "max_iter": 5000,
    "M_divergence": 1000.0,
    "_4": "Plotting parameters: base field",
    "base_layer": "continuous_iter",
    "colormap": fs.colors.Fractal_colormap(
        colors=[[0.00000000e+00, 2.74509804e-02, 3.92156863e-01],
             [1.10437384e-04, 3.73834922e-02, 4.04650357e-01],
             [4.38226764e-04, 4.73039863e-02, 4.17125524e-01],
             [9.78083979e-04, 5.72121456e-02, 4.29573705e-01],
             [1.72472487e-03, 6.71076526e-02, 4.41986240e-01],
             [2.67286527e-03, 7.69901900e-02, 4.54354469e-01],
             [3.81722103e-03, 8.68594405e-02, 4.66669733e-01],
             [5.15250798e-03, 9.67150866e-02, 4.78923371e-01],
             [6.67344196e-03, 1.06556811e-01, 4.91106726e-01],
             [8.37473881e-03, 1.16384297e-01, 5.03211136e-01],
             [1.02511144e-02, 1.26197226e-01, 5.15227942e-01],
             [1.22972845e-02, 1.35995282e-01, 5.27148486e-01],
             [1.45079650e-02, 1.45778147e-01, 5.38964106e-01],
             [1.68778717e-02, 1.55545503e-01, 5.50666144e-01],
             [1.94017206e-02, 1.65297034e-01, 5.62245939e-01],
             [2.20742273e-02, 1.75032422e-01, 5.73694833e-01],
             [2.48901077e-02, 1.84751350e-01, 5.85004166e-01],
             [2.78440778e-02, 1.94453500e-01, 5.96165278e-01],
             [3.09308533e-02, 2.04138555e-01, 6.07169509e-01],
             [3.41451500e-02, 2.13806197e-01, 6.18008200e-01],
             [3.74816839e-02, 2.23456110e-01, 6.28672691e-01],
             [4.09351707e-02, 2.33087977e-01, 6.39154323e-01],
             [4.45003263e-02, 2.42701478e-01, 6.49444436e-01],
             [4.81718665e-02, 2.52296298e-01, 6.59534371e-01],
             [5.19445072e-02, 2.61872119e-01, 6.69415467e-01],
             [5.58129642e-02, 2.71428624e-01, 6.79079066e-01],
             [5.97719533e-02, 2.80965495e-01, 6.88516507e-01],
             [6.38161904e-02, 2.90482415e-01, 6.97719131e-01],
             [6.79403913e-02, 2.99979067e-01, 7.06678279e-01],
             [7.21392719e-02, 3.09455133e-01, 7.15385291e-01],
             [7.64075480e-02, 3.18910296e-01, 7.23831507e-01],
             [8.07399355e-02, 3.28344239e-01, 7.32008267e-01],
             [8.51311501e-02, 3.37756644e-01, 7.39906913e-01],
             [8.95759078e-02, 3.47147194e-01, 7.47518784e-01],
             [9.40689243e-02, 3.56515572e-01, 7.54835221e-01],
             [9.86049155e-02, 3.65861460e-01, 7.61847564e-01],
             [1.03178597e-01, 3.75184541e-01, 7.68547154e-01],
             [1.07784685e-01, 3.84484498e-01, 7.74925331e-01],
             [1.12417896e-01, 3.93761012e-01, 7.80973436e-01],
             [1.17072944e-01, 4.03013768e-01, 7.86682808e-01],
             [1.21744546e-01, 4.12242448e-01, 7.92044788e-01],
             [1.26442213e-01, 4.21451959e-01, 7.97057643e-01],
             [1.31644451e-01, 4.30811773e-01, 8.01942937e-01],
             [1.37566242e-01, 4.40395326e-01, 8.06809033e-01],
             [1.44183896e-01, 4.50191275e-01, 8.11654219e-01],
             [1.51473724e-01, 4.60188274e-01, 8.16476784e-01],
             [1.59412036e-01, 4.70374980e-01, 8.21275017e-01],
             [1.67975141e-01, 4.80740047e-01, 8.26047207e-01],
             [1.77139351e-01, 4.91272131e-01, 8.30791641e-01],
             [1.86880974e-01, 5.01959888e-01, 8.35506611e-01],
             [1.97176323e-01, 5.12791973e-01, 8.40190402e-01],
             [2.08001706e-01, 5.23757041e-01, 8.44841306e-01],
             [2.19333434e-01, 5.34843749e-01, 8.49457610e-01],
             [2.31147817e-01, 5.46040752e-01, 8.54037602e-01],
             [2.43421166e-01, 5.57336705e-01, 8.58579573e-01],
             [2.56129790e-01, 5.68720263e-01, 8.63081810e-01],
             [2.69250000e-01, 5.80180083e-01, 8.67542602e-01],
             [2.82758106e-01, 5.91704820e-01, 8.71960239e-01],
             [2.96630418e-01, 6.03283129e-01, 8.76333008e-01],
             [3.10843247e-01, 6.14903666e-01, 8.80659199e-01],
             [3.25372903e-01, 6.26555086e-01, 8.84937100e-01],
             [3.40195695e-01, 6.38226045e-01, 8.89165000e-01],
             [3.55287934e-01, 6.49905198e-01, 8.93341187e-01],
             [3.70625931e-01, 6.61581201e-01, 8.97463951e-01],
             [3.86185996e-01, 6.73242710e-01, 9.01531580e-01],
             [4.01944438e-01, 6.84878379e-01, 9.05542364e-01],
             [4.17877568e-01, 6.96476865e-01, 9.09494590e-01],
             [4.33961696e-01, 7.08026823e-01, 9.13386547e-01],
             [4.50173133e-01, 7.19516908e-01, 9.17216524e-01],
             [4.66488189e-01, 7.30935776e-01, 9.20982811e-01],
             [4.82883173e-01, 7.42272083e-01, 9.24683695e-01],
             [4.99334396e-01, 7.53514484e-01, 9.28317465e-01],
             [5.15818169e-01, 7.64651634e-01, 9.31882411e-01],
             [5.32310801e-01, 7.75672189e-01, 9.35376820e-01],
             [5.48788603e-01, 7.86564805e-01, 9.38798982e-01],
             [5.65227885e-01, 7.97318137e-01, 9.42147185e-01],
             [5.81604957e-01, 8.07920840e-01, 9.45419719e-01],
             [5.97896129e-01, 8.18361570e-01, 9.48614871e-01],
             [6.14077712e-01, 8.28628983e-01, 9.51730931e-01],
             [6.30126016e-01, 8.38711735e-01, 9.54766187e-01],
             [6.46017351e-01, 8.48598479e-01, 9.57718928e-01],
             [6.61728027e-01, 8.58277873e-01, 9.60587443e-01],
             [6.77234354e-01, 8.67738572e-01, 9.63370021e-01],
             [6.92512643e-01, 8.76969230e-01, 9.66064950e-01],
             [7.07539204e-01, 8.85958505e-01, 9.68670518e-01],
             [7.22290347e-01, 8.94695050e-01, 9.71185016e-01],
             [7.36742383e-01, 9.03167523e-01, 9.73606731e-01],
             [7.50871621e-01, 9.11364577e-01, 9.75933952e-01],
             [7.64654371e-01, 9.19274869e-01, 9.78164968e-01],
             [7.78066945e-01, 9.26887055e-01, 9.80298068e-01],
             [7.91085652e-01, 9.34189789e-01, 9.82331541e-01],
             [8.03686802e-01, 9.41171728e-01, 9.84263674e-01],
             [8.15846706e-01, 9.47821526e-01, 9.86092758e-01],
             [8.27541674e-01, 9.54127840e-01, 9.87817080e-01],
             [8.38748016e-01, 9.60079324e-01, 9.89434930e-01],
             [8.49442042e-01, 9.65664635e-01, 9.90944596e-01],
             [8.59600062e-01, 9.70872428e-01, 9.92344366e-01],
             [8.69198387e-01, 9.75691358e-01, 9.93632531e-01],
             [8.78213328e-01, 9.80110081e-01, 9.94807377e-01],
             [8.86621193e-01, 9.84117253e-01, 9.95867195e-01],
             [8.94398294e-01, 9.87701528e-01, 9.96810273e-01],
             [9.01520940e-01, 9.90851563e-01, 9.97634899e-01],
             [9.07965442e-01, 9.93556012e-01, 9.98339363e-01],
             [9.13708110e-01, 9.95803532e-01, 9.98921953e-01],
             [9.18725254e-01, 9.97582779e-01, 9.99380957e-01],
             [9.22993185e-01, 9.98882406e-01, 9.99714665e-01],
             [9.26488212e-01, 9.99691071e-01, 9.99921366e-01],
             [9.29186646e-01, 9.99997428e-01, 9.99999347e-01],
             [9.31390967e-01, 9.99862137e-01, 9.99253124e-01],
             [9.33566587e-01, 9.99389882e-01, 9.96710858e-01],
             [9.35716976e-01, 9.98588684e-01, 9.92429575e-01],
             [9.37841616e-01, 9.97465705e-01, 9.86474975e-01],
             [9.39939988e-01, 9.96028108e-01, 9.78912758e-01],
             [9.42011573e-01, 9.94283054e-01, 9.69808626e-01],
             [9.44055854e-01, 9.92237706e-01, 9.59228278e-01],
             [9.46072311e-01, 9.89899227e-01, 9.47237417e-01],
             [9.48060428e-01, 9.87274778e-01, 9.33901742e-01],
             [9.50019684e-01, 9.84371523e-01, 9.19286955e-01],
             [9.51949561e-01, 9.81196622e-01, 9.03458756e-01],
             [9.53849542e-01, 9.77757239e-01, 8.86482847e-01],
             [9.55719108e-01, 9.74060536e-01, 8.68424927e-01],
             [9.57557740e-01, 9.70113676e-01, 8.49350698e-01],
             [9.59364920e-01, 9.65923819e-01, 8.29325861e-01],
             [9.61140129e-01, 9.61498130e-01, 8.08416116e-01],
             [9.62882850e-01, 9.56843770e-01, 7.86687164e-01],
             [9.64592563e-01, 9.51967901e-01, 7.64204706e-01],
             [9.66268750e-01, 9.46877686e-01, 7.41034443e-01],
             [9.67910894e-01, 9.41580287e-01, 7.17242075e-01],
             [9.69518474e-01, 9.36082866e-01, 6.92893304e-01],
             [9.71090974e-01, 9.30392586e-01, 6.68053830e-01],
             [9.72627874e-01, 9.24516609e-01, 6.42789353e-01],
             [9.74128656e-01, 9.18462098e-01, 6.17165576e-01],
             [9.75592802e-01, 9.12236214e-01, 5.91248198e-01],
             [9.77019794e-01, 9.05846120e-01, 5.65102920e-01],
             [9.78409112e-01, 8.99298978e-01, 5.38795444e-01],
             [9.79760238e-01, 8.92601951e-01, 5.12391470e-01],
             [9.81072655e-01, 8.85762201e-01, 4.85956698e-01],
             [9.82345843e-01, 8.78786890e-01, 4.59556830e-01],
             [9.83579285e-01, 8.71683180e-01, 4.33257566e-01],
             [9.84772461e-01, 8.64458234e-01, 4.07124607e-01],
             [9.85924854e-01, 8.57119214e-01, 3.81223655e-01],
             [9.87035944e-01, 8.49673283e-01, 3.55620409e-01],
             [9.88105214e-01, 8.42127602e-01, 3.30380570e-01],
             [9.89132146e-01, 8.34489335e-01, 3.05569840e-01],
             [9.90116220e-01, 8.26765643e-01, 2.81253919e-01],
             [9.91056918e-01, 8.18963688e-01, 2.57498508e-01],
             [9.91953722e-01, 8.11090634e-01, 2.34369308e-01],
             [9.92806113e-01, 8.03153642e-01, 2.11932020e-01],
             [9.93613573e-01, 7.95159874e-01, 1.90252343e-01],
             [9.94375584e-01, 7.87116493e-01, 1.69395980e-01],
             [9.95091627e-01, 7.79030662e-01, 1.49428631e-01],
             [9.95761184e-01, 7.70909542e-01, 1.30415997e-01],
             [9.96383736e-01, 7.62760296e-01, 1.12423778e-01],
             [9.96958765e-01, 7.54590086e-01, 9.55176752e-02],
             [9.97485753e-01, 7.46406074e-01, 7.97633899e-02],
             [9.97964180e-01, 7.38215424e-01, 6.52266225e-02],
             [9.98393529e-01, 7.30025296e-01, 5.19730738e-02],
             [9.98773282e-01, 7.21842854e-01, 4.00684447e-02],
             [9.99102919e-01, 7.13675259e-01, 2.95784359e-02],
             [9.99381923e-01, 7.05529675e-01, 2.05687483e-02],
             [9.99609774e-01, 6.97413263e-01, 1.31050827e-02],
             [9.99785955e-01, 6.89333185e-01, 7.25313979e-03],
             [9.99909947e-01, 6.81296605e-01, 3.07862044e-03],
             [9.99981232e-01, 6.73310683e-01, 6.47225458e-04],
             [9.99973697e-01, 6.65372012e-01, 0.00000000e+00],
             [9.98670259e-01, 6.56984091e-01, 0.00000000e+00],
             [9.95455321e-01, 6.47895389e-01, 0.00000000e+00],
             [9.90401699e-01, 6.38138096e-01, 0.00000000e+00],
             [9.83582215e-01, 6.27744402e-01, 0.00000000e+00],
             [9.75069687e-01, 6.16746500e-01, 0.00000000e+00],
             [9.64936933e-01, 6.05176579e-01, 0.00000000e+00],
             [9.53256774e-01, 5.93066831e-01, 0.00000000e+00],
             [9.40102027e-01, 5.80449446e-01, 0.00000000e+00],
             [9.25545513e-01, 5.67356615e-01, 0.00000000e+00],
             [9.09660051e-01, 5.53820529e-01, 0.00000000e+00],
             [8.92518458e-01, 5.39873379e-01, 0.00000000e+00],
             [8.74193556e-01, 5.25547356e-01, 0.00000000e+00],
             [8.54758162e-01, 5.10874650e-01, 0.00000000e+00],
             [8.34285096e-01, 4.95887452e-01, 0.00000000e+00],
             [8.12847176e-01, 4.80617953e-01, 0.00000000e+00],
             [7.90517223e-01, 4.65098344e-01, 0.00000000e+00],
             [7.67368055e-01, 4.49360816e-01, 0.00000000e+00],
             [7.43472491e-01, 4.33437559e-01, 0.00000000e+00],
             [7.18903350e-01, 4.17360764e-01, 0.00000000e+00],
             [6.93733452e-01, 4.01162623e-01, 0.00000000e+00],
             [6.68035615e-01, 3.84875326e-01, 0.00000000e+00],
             [6.41882659e-01, 3.68531063e-01, 0.00000000e+00],
             [6.15347402e-01, 3.52162026e-01, 0.00000000e+00],
             [5.88502665e-01, 3.35800405e-01, 0.00000000e+00],
             [5.61421265e-01, 3.19478392e-01, 0.00000000e+00],
             [5.34176022e-01, 3.03228177e-01, 0.00000000e+00],
             [5.06839756e-01, 2.87081950e-01, 0.00000000e+00],
             [4.79485284e-01, 2.71071903e-01, 0.00000000e+00],
             [4.52185427e-01, 2.55230227e-01, 0.00000000e+00],
             [4.25013004e-01, 2.39589112e-01, 0.00000000e+00],
             [3.98040832e-01, 2.24180749e-01, 0.00000000e+00],
             [3.71341733e-01, 2.09037330e-01, 0.00000000e+00],
             [3.44988524e-01, 1.94191044e-01, 0.00000000e+00],
             [3.19054025e-01, 1.79674082e-01, 0.00000000e+00],
             [2.93611055e-01, 1.65518636e-01, 0.00000000e+00],
             [2.68732434e-01, 1.51756896e-01, 0.00000000e+00],
             [2.44490979e-01, 1.38421054e-01, 0.00000000e+00],
             [2.20959510e-01, 1.25543299e-01, 0.00000000e+00],
             [1.98210847e-01, 1.13155823e-01, 0.00000000e+00],
             [1.76317808e-01, 1.01290816e-01, 0.00000000e+00],
             [1.55353213e-01, 8.99804696e-02, 0.00000000e+00],
             [1.35389881e-01, 7.92569745e-02, 0.00000000e+00],
             [1.16500630e-01, 6.91525213e-02, 0.00000000e+00],
             [9.87582805e-02, 5.96993009e-02, 0.00000000e+00],
             [8.22356507e-02, 5.09295041e-02, 0.00000000e+00],
             [6.70055599e-02, 4.28753217e-02, 0.00000000e+00],
             [5.31408274e-02, 3.55689446e-02, 0.00000000e+00],
             [4.07142721e-02, 2.90425636e-02, 0.00000000e+00],
             [2.97987132e-02, 2.33283694e-02, 0.00000000e+00],
             [2.04669698e-02, 1.84585530e-02, 0.00000000e+00],
             [1.27918610e-02, 1.44653050e-02, 0.00000000e+00],
             [6.84620587e-03, 1.13808164e-02, 0.00000000e+00],
             [2.70282357e-03, 9.23727801e-03, 0.00000000e+00],
             [4.34533158e-04, 8.06688056e-03, 0.00000000e+00],
             [0.00000000e+00, 7.84315297e-03, 5.39857630e-05],
             [0.00000000e+00, 7.84411505e-03, 8.39057327e-04],
             [0.00000000e+00, 7.84835664e-03, 2.53589902e-03],
             [0.00000000e+00, 7.85832973e-03, 5.11501856e-03],
             [0.00000000e+00, 7.87648628e-03, 8.54692366e-03],
             [0.00000000e+00, 7.90527827e-03, 1.28021221e-02],
             [0.00000000e+00, 7.94715768e-03, 1.78511215e-02],
             [0.00000000e+00, 8.00457647e-03, 2.36644296e-02],
             [0.00000000e+00, 8.07998662e-03, 3.02125542e-02],
             [0.00000000e+00, 8.17584011e-03, 3.74660029e-02],
             [0.00000000e+00, 8.29458892e-03, 4.53952835e-02],
             [0.00000000e+00, 8.43868501e-03, 5.39709038e-02],
             [0.00000000e+00, 8.61058036e-03, 6.31633713e-02],
             [0.00000000e+00, 8.81272694e-03, 7.29431939e-02],
             [0.00000000e+00, 9.04757673e-03, 8.32808793e-02],
             [0.00000000e+00, 9.31758171e-03, 9.41469352e-02],
             [0.00000000e+00, 9.62519385e-03, 1.05511869e-01],
             [0.00000000e+00, 9.97286511e-03, 1.17346189e-01],
             [0.00000000e+00, 1.03630475e-02, 1.29620403e-01],
             [0.00000000e+00, 1.07981929e-02, 1.42305018e-01],
             [0.00000000e+00, 1.12807535e-02, 1.55370542e-01],
             [0.00000000e+00, 1.18131810e-02, 1.68787483e-01],
             [0.00000000e+00, 1.23979275e-02, 1.82526348e-01],
             [0.00000000e+00, 1.30374451e-02, 1.96557646e-01],
             [0.00000000e+00, 1.37341856e-02, 2.10851884e-01],
             [0.00000000e+00, 1.44906010e-02, 2.25379569e-01],
             [0.00000000e+00, 1.53091433e-02, 2.40111210e-01],
             [0.00000000e+00, 1.61922645e-02, 2.55017314e-01],
             [0.00000000e+00, 1.71424165e-02, 2.70068388e-01],
             [0.00000000e+00, 1.81620514e-02, 2.85234942e-01],
             [0.00000000e+00, 1.92536210e-02, 3.00487481e-01],
             [0.00000000e+00, 2.04195775e-02, 3.15796515e-01],
             [0.00000000e+00, 2.16623727e-02, 3.31132550e-01],
             [0.00000000e+00, 2.29844586e-02, 3.46466095e-01],
             [0.00000000e+00, 2.43882872e-02, 3.61767657e-01],
             [0.00000000e+00, 2.58763105e-02, 3.77007743e-01],
             [0.00000000e+00, 2.74509804e-02, 3.92156863e-01]],
        kinds=['Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch',
             'Lch', 'Lch'],
        grad_npts=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3],
        grad_funcs=['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',
             'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        extent='repeat'
),
    "cmap_func": fractalshades.numpy_utils.expr_parser.Numpy_expr(
        variables=[
            "x",
        ],
        expr="x",
    ),
    "zmin": 0.0,
    "zmax": 10.0,
    "zshift": 1.0,
    "mask_color": (
        0.1,
        0.1,
        0.1,
        1.0,
    ),
    "_5": "Plotting parameters: shading",
    "has_shading": False,
    "shading_kind": "potential",
    "lighting": fs.colors.layers.Blinn_lighting(
        k_ambient=0.4,
        color_ambient=[1., 1., 1.],
        ls0={
            'k_diffuse': 1.8,
            'k_specular': 15.0,
            'shininess': 500.0,
            'polar_angle': 50.0,
            'azimuth_angle': 20.0,
            'color': [1.  , 1.  , 0.95],
            'material_specular_color': None
        },
    ),
    "max_slope": 60.0,
    "_6": "Plotting parameters: field lines",
    "has_fieldlines": False,
    "fieldlines_func": fractalshades.numpy_utils.expr_parser.Numpy_expr(
        variables=[
            "x",
        ],
        expr="x",
    ),
    "fieldlines_kind": "twin",
    "fieldlines_zmin": -1.0,
    "fieldlines_zmax": 1.0,
    "backshift": 4,
    "n_iter": 4,
    "swirl": 0.0,
    "damping_ratio": 0.8,
    "twin_intensity": 0.5,
    "_8": "High-quality rendering options",
    "final_render": False,
    "supersampling": "3x3",
    "jitter": False,
    "recovery_mode": False,
    "_9": "Extra outputs",
    "output_masks": False,
    "output_normals": False,
    "output_heightmaps": False,
    "hmap_mask": 0.0,
    "int_hmap_mask": 0.0,
    "_10": "General settings",
    "log_verbosity": "debug @ console + log",
    "enable_multithreading": True,
    "inspect_calc": False,
    "no_newton": False,
    "postproc_dtype": "float32",
    "compute_newton": False,
    "_3": None,
    "max_order": None,
    "max_newton": None,
    "eps_newton_cv": None,
    "_7": None,
    "int_layer": None,
    "colormap_int": None,
    "cmap_func_int": None,
    "zmin_int": None,
    "zmax_int": None,
    "lighting_int": None,
    "calc_dzndc": False,
    "interior_detect": False,
    "epsilon_stationnary": None,
}

#------------------------------------------------------------------------------
# Function - /!\ do not modify this section
#------------------------------------------------------------------------------
def get_plotter(
    fractal: fs.Fractal= fractalshades.models.burning_ship.Perturbation_burning_ship(
        directory=plot_dir,
        flavor="Burning ship",
    ),
    calc_name: str="std_zooming_calc",

    _1: fs.gui.collapsible_separator="Zoom parameters",
    x : mpmath.mpf = "0.0",
    y : mpmath.mpf = "0.0",
    dx : mpmath.mpf = "10.0",
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
    interior_detect: bool = False,
    epsilon_stationnary: float = None,
    calc_dzndc: bool = False,

    _3: fs.gui.collapsible_separator = None,
    compute_newton: bool = False,
    max_order: int = None,
    max_newton: int = None,
    eps_newton_cv: float = None,

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

    _7: fs.gui.collapsible_separator= None,
    int_layer: typing.Literal[
        "attractivity", "order", "attr / order"
    ]= None,
    colormap_int: fs.colors.Fractal_colormap = None,
    cmap_func_int: fs.numpy_utils.Numpy_expr = None,
    zmin_int: float = None,
    zmax_int: float = None,

    _5: fs.gui.collapsible_separator = "Plotting parameters: shading",
    has_shading: bool = True,
    shading_kind: typing.Literal["potential"] = "potential", 
    lighting: Blinn_lighting = (
            fs.colors.lighting_register["glossy"]
    ),
    lighting_int: Blinn_lighting = None,
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
    postproc_dtype: typing.Literal["float32", "float64"] = "float32",
    batch_params=batch_params
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
        "has_skew": has_skew,
        "skew_00": skew_00,
        "skew_01": skew_01,
        "skew_10": skew_10,
        "skew_11": skew_11,
        "projection": batch_params.get(
            "projection", fs.projection.Cartesian()
        )
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

    
    return plotter, plotter[base_layer].postname
        
