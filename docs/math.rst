Mathematical background
***********************

For a basic introduction to the Mandelbrot set we recommend this wiki_
page by Arnaud Chéritat.

This section will only present a brief outline of the algorithms used in
fractalshades for deep Mandelbrot zooms. Do not expect rigorous
demonstrations. For a more in-depth coverage of these points you should
also have a look at Claude Heiland-Allen blog_, particularly these entries:

  - https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html
  - https://mathr.co.uk/blog/2022-02-21_deep_zoom_theory_and_practice_again.html


Zoom level and native floating-point precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recall the basic iterations fo the Mandelbrot set:

.. math::

    \left\{
    \begin{array}{rcl}
    z_0 &= &0 \\
    z_{n+1} &= &z_{n}^2 + c
    \end{array}
    \right.

Here, :math:`c` represents a pixel in the image. For all interesting
areas in the Mandelbrot set, the magnitude of :math:`|c|` is close to
:math:`1.0` - somewhere between :math:`0.5` and :math:`2.0`. 

On most computers the highest precision floating-point format available
natively is double_, with a significand precision of 53 bits. Its means that,
at machine precision we have

.. code-block:: console

    > 1. + 2 ** -53 == 1.
    True

So for 2 pixels separated by less than :math:`2^{-53}`, a calulation at the
highest available machine precision will result in exactly the same sequence.
This is the maximal resolution for doubles, roughly :math:`10^{-16}`.
If the image is 1000 pixels width, issues begins to occur for a width
of :math:`10^{-13}`.

To go deeper one need to switch to an arbitrary-precision library (like
gmpy2_).

Unfortunately calculations will become a lot slower,
and moreover this time will increase quickly with the zoom factor.
Algorithms have been developped to avoid running all pixels in full precision
(in fact, one orbit calculated at full precision and stored at standard
precision if often enough)


Perturbation Theory
~~~~~~~~~~~~~~~~~~~~

This technique is based on an original idea from  K. I. Martin (see 
:cite:p:`superfractalthing`)

Let's assume we know a reference sequence :math:`Z_n` (for instance, the
center of the image iterated with arbitrary precision):

.. math::

    \left\{
    \begin{array}{rcl}
    Z_0 &= &0 \\
    Z_{n+1} &= &Z_{n}^2 + C
    \end{array}
    \right.

For any point :math:`c` in the image we can express it as
:math:`c = C + \delta c` where :math:`\delta c` is small.
Similarly for its iterated sequence we will write:
:math:`z_n = Z_n + \delta z_n`.

Some elementary manipulations yield the recurrence formula for
:math:`\delta z_n` :

.. math::
    :label: perturb

    \left\{
    \begin{array}{rcl}
    \delta z_0 &= &0 \\
    \delta z_{n+1} &= & \delta z_n (2 Z_n + \delta z_n) + \delta c 
    \end{array}
    \right.

Similar formulas can be found for the derivatives (useful for instance
for distance approximation, or early interior points detection)
:math:`\frac {\partial \delta z_n}{\partial c}` and
:math:`\frac {\partial \delta z_n}{\partial z_1}`.

The good and a bit unexpected news is that :math:`\delta z_n` can be iterated
at standard precision and, with a few caveat, still give similar results to a
calculation with full precision! A rigorous demonstation is
probably quite involving and in any case beyond the scope of this section... 


Avoiding loss of precision
~~~~~~~~~~~~~~~~~~~~~~~~~~

When iterating :eq:`perturb` numerical loss of precision may occur,
leading to invalid results. This is often evidenced by so-called "glitches"
in the image: large zones with uniform color or noisy rendering.

A deeper investigation shows that these issues happen when the calculated
orbit :math:`z_n` goes close to zero [#f1]_, under the influence of a nearby
minibrot of period :math:`n`.

Several methods have been investigated to avoid such glitches.

We present here the one used in `fractalshades` which is based
on an original idea from Zhuoran in 2021 (see :cite:p:`glitches`). Whenever 
:math:`|z_n| <= |Z_n|` we do a *rebase* i.e. basically restart the reference
orbit at index 0:

.. math::
    :label: rebase

    \left\{
    \begin{array}{rcll}
    \delta z_{n+1} &\leftarrow &z_n & \\
    Z_{n+1+i} &\leftarrow &Z_i &\forall i \in \mathbb{N}
    \end{array}
    \right.

Of course from an implementation point of view it is easier to just
reset the index :math:`n` in :math:`Z_n` to 0 - rather than making the
substitution above.

The same strategy can also be applied whenever the reference orbit diverges.

Billinear approximations
~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`|\delta z_n| \ll |2 Z_n|`,
:eq:`perturb` can be approximated without loss of precision by
a billinear expression in :math:`\delta z_n` and :math:`\delta c` :

.. math::

    \delta z_{n+1} = 2 Z_n \delta z_n + \delta c 
                   = A_n \delta z_n + B_n \delta c

Or, with more precise notation highlighting that this set of billinear
coefficients map :math:`\delta z_n` to :math:`\delta z_{n+1}`:

.. math::
    :label: perturb_BLA

    \delta z_{n+1} = A_{n, n+1} \cdot \delta z_n + B_{n, n+1} \cdot \delta c 
    \quad \forall \, \delta z_n, \, |\delta z_n| \ll |2 Z_n| = r_{n, n+1}

Now, the composition of 2 bilinear functions is still a billinear function.
Combining 2 by 2 gives:

.. math::
    :label: composition

    \left\{
    \begin{array}{rcll}
    A_{k, m} &= & A_{k, l} \cdot A_{l, m}& \\
    B_{k, m} &= & B_{k, l} \cdot A_{l, m} + A_{l, m} 
    \end{array}
    \right.

Valid if:

.. math::

    \begin{array}{rcl}
    |\delta z_n| &\leq & \min \left( r_{k, l}\, , \: 
    \max \left( 0\, , \:
    r_{l, m} - \frac{|A_{k, l}| |\delta c|}{|B_{k, l}|}
    \right) \right) \\
     &\leq & r_{k, m}
    \end{array}

It means, we can combine 2 by 2 such approximations (and their validity bound
:math:`r_{k, m}`) and pre-compute the results - for instance in a binary tree
structure.
This gives a set of precomputed *shortcuts* allowing to skip many iterations,
when the validity conditions are met.

Note: The same approach can also be applied to the derivatives,
they just becomes a simple linear relationship.

Extended range floating points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another limitation of native double_ format is met when the size of a pixel
drops below approx :math:`10^{-308}`. This time, it is the range of the
exponent.

For IEEE 754 double, exponent has 11 bits and its valid range is :math:`-1022`
to :math:`+1023`. Numbers below :math:`2^{-1022}` are first stored by
compromising precision (subnormal numbers), down to about
:math:`5 \cdot 10^{-324}` and then they vanish:

.. code-block:: console

    > 10. ** -324 == 0.
    True

In short, manipulating numbers below :math:`10^{-308}` in double format shall
be avoided. `fractalshades` switches to a ad-hoc datatype based on a couple
(float64, int32) and the main operations have been implemented for this
datatype.

Finding minibrots
~~~~~~~~~~~~~~~~~

In order to find a minibrot center and size, the process is the following:

  - find the period first by iterating a small ball until it contains the
    origin (using ball arithmetic),

  - then use Newton’s method to find the location of the minibrot with the
    computed period

  - then compute the size (and orientation) using a renormalization formula
    (as the one found in :cite:p:`size_estimate2`)

We will not describe in details here these techniques which have already been
better explained elsewhere. The interested reader might refer to: 

  - :cite:cts:`mandelbrot_notes`
  - :cite:cts:`size_estimate`.

A generalisation of these techniques for 
the *Burning Ship* is described in :cite:t:`burning_ship`.

An implementation of these algorithms is available from the GUI (right-clik
on the image) for the fractals supporting arbitrary-precision.

Going further: a few references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. bibliography::
   :all:


.. _wiki: https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
.. _blog: https://mathr.co.uk/blog/
.. _double: https://en.wikipedia.org/wiki/Double-precision_floating-point_format
.. _gmpy2: https://pypi.org/project/gmpy2/


.. [#f1] Or more generally, a critical point...


