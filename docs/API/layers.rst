Image layers
~~~~~~~~~~~~

Fractalshades graphical export is based on the concept of layers.
Each layer is linked to a postprocessing field, it can be directly plotted
or can be used to modify another layer.

.. module:: fractalshades.colors.layers

.. autoclass:: fractalshades.colors.layers.Virtual_layer
    :members: __init__, set_mask

.. autoclass:: fractalshades.colors.layers.Bool_layer
    :members:  __init__

.. autoclass:: fractalshades.colors.layers.Color_layer
    :members:  __init__, set_mask, shade, overlay, set_twin_field

.. autoclass:: fractalshades.colors.layers.Normal_map_layer
    :members: __init__, set_mask

.. autoclass:: fractalshades.colors.layers.Grey_layer
    :members:  __init__, set_mask

.. autoclass:: fractalshades.colors.layers.Disp_layer
    :members:  __init__, set_mask

.. autoclass:: fractalshades.colors.layers.Blinn_lighting
    :members: __init__, add_light_source

.. autoclass:: fractalshades.colors.layers.Overlay_mode
    :members: __init__
