Image layers
~~~~~~~~~~~~

Fractalshades graphical export is based on the concept of layers.
Each layer is linked to a postprocessing field, it can be directly plotted
or can be used to modify another layer.

.. module:: fractalshades.colors.layers

.. autoclass:: Virtual_layer
    :members: __init__, set_mask

.. autoclass:: Bool_layer
    :members:  __init__

.. autoclass:: Color_layer
    :members:  __init__, set_mask, shade, overlay, set_twin_field 

.. autoclass:: Normal_map_layer

.. autoclass:: Grey_layer
    :members:  __init__, set_mask

.. autoclass:: Disp_Layer
    :members:

.. autoclass:: Blinn_lighting
    :members: __init__, add_light_source

.. autoclass:: Overlay_mode
    :members: __init__
