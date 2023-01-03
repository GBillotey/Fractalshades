.. module:: fractalshades.models
    :noindex:

Fractal models: Standard-precision implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard precision implementations are limited by ``float64`` format
precision.


.. autoclass:: fractalshades.models.Mandelbrot
    :members: __init__, calc_std_div, newton_calc

.. autoclass:: fractalshades.models.Mandelbrot_N
    :members: __init__, calc_std_div, newton_calc

.. autoclass:: fractalshades.models.Burning_ship
    :members: __init__, calc_std_div

.. autoclass:: fractalshades.models.Power_tower
    :members: __init__, newton_calc

.. autoclass:: fractalshades.models.Collatz
    :members: __init__, base_calc

