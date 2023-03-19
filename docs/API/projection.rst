Projections
~~~~~~~~~~~

This section describes the projections (pixel mappings) availables.

.. note::

    Only the Cartesian projection is available in the GUI. To use other
    projections, export a script from the GUI (once coloring options are
    selected...)
    and modify the `projection` parameter in batch mode:

    .. code::

        batch_params = {"projection": projection}


.. module:: fractalshades.projection

.. autoclass:: fractalshades.projection.Projection
    :members: __init__

.. autoclass:: fractalshades.projection.Cartesian
    :members: __init__

.. autoclass:: fractalshades.projection.Expmap
    :members: __init__

.. autoclass:: fractalshades.projection.Generic_mapping
    :members: __init__



