Extending
*********

This section provides details for the advanced used on how to 
extend fractalshades.

new coloring technique
~~~~~~~~~~~~~~~~~~~~~~

To add a new coloring technique, one should first consider subclassing
`fractalshades.postproc.Postproc`. If the needed raw complex or integer
fields are already available, this will be enough.

If a new calculation output is needed, it will be necessary to also add a
new calculation routine to one of the `fractalshades.Fractal` models.
This would be the case for instance if you wish to add a custom orbit trap.
This can either be done through sub-classing or monkey-patching. Remember to
tag (decorate) your routine with @ `fractalshades.calc_options`.

new fractal
~~~~~~~~~~~

To define a new fracal, subclass `fractalshades.Fractal`. This is almost
straightforward for standard (double-precision) fractals as one only has to 
follow the steps :

- Defines the list of complex codes (corresponding to the complex fields
  passed for each point at each iteration), integer codes (the integer fields)
  and stop codes (the reason for exiting the loop) ;
  store them as class attributes

  .. code-block:: python

      self.codes = (complex_codes, int_codes, stop_codes)


- Initialise the datatypes, usually with call to `init_data_types`:

  .. code-block:: python

      self.init_data_types(np.complex128)

- Define the initialisation function and store it as a class attribute (Note
  that we use a wrapper to allow customization). This step is done with 
  classic numpy calls.
  As an example if we only have 4 complex fields
  [“zn”, “dzndz”, “dzndc”, “d2zndc2”] and no integer field:

  .. code-block:: python

        def initialize():
            def func(Z, U, c, chunk_slice):
                Z[3, :] = 0.
                Z[2, :] = 0.
                Z[1, :] = 0.
                Z[0, :] = 0.
            return func
        self.initialize = initialize

- Define the iteration function and store it as a class attribute (Note
  that we use a wrapper to allow customization). This step is time
  critical so is done with numba jitted functions (njit).

  As an example if we only have 4 complex fields 
  ["zn", "dzndz", "dzndc", "d2zndc2"] and no integer field:

  .. code-block:: python

        def iterate():
            Mdiv_sq = self.M_divergence ** 2
            epscv_sq = self.epsilon_stationnary ** 2
            max_iter = self.max_iter
            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):
                if n_iter >= max_iter:
                    stop_reason[0] = 0

                Z[3] = 2 * (Z[3] * Z[0] + Z[2]**2)
                Z[2] = 2 * Z[2] * Z[0] + 1.
                Z[1] = 2 * Z[1] * Z[0] 
                Z[0] = Z[0]**2 + c
                if n_iter == 1:
                    Z[1] = 1.

                if Z[0].real ** 2 + Z[0].imag ** 2 > Mdiv_sq:
                    stop_reason[0] = 1
                if Z[1].real ** 2 + Z[1].imag ** 2 < epscv_sq:
                    stop_reason[0] = 2

            return numba_impl
        self.iterate = iterate

- Remember that inner loop will exit whenever stop_reason[0] is set to a
  value != -1. The value shall be a null or positive integer : 
  corresponding to the index in `stop_codes`. For instance with `stop_codes`
  ["max_iter", "divergence", "stationnary"] if you reach the criteria for
  divergence you have noticed line :

  .. code-block:: python

        if Z[0].real ** 2 + Z[0].imag ** 2 > Mdiv_sq:
            stop_reason[0] = 1

Et voila (seee also `fractalshades.models.Mandelbrot.base_calc` source code).
To be adapted for your needs...

For fractals implementing perturbation technique, this is beyong the scope
of this use manual.


