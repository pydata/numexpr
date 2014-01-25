==================
Installing Numexpr
==================

These are instructions for installing Numexpr on Unix systems.  For
Windows, it is best to install it from binaries.  However, you should
note that, for the time being, we cannot provide Windows binaries with
MKL support.


Building
========

This version of `Numexpr` requires Python 2.6 or greater,
and NumPy 1.6 or greater.

It's built in the standard Python way::

  $ python setup.py build
  $ python setup.py install

You can test `numexpr` with:

  $ python -c "import numexpr; numexpr.test()"


Enabling Intel's MKL support
============================

numexpr includes support for Intel's MKL library.  This allows for
better performance on Intel architectures, mainly when evaluating
transcendental functions (trigonometrical, exponential...).  It also
enables numexpr using several CPU cores.

If you have Intel's MKL, just copy the `site.cfg.example` that comes
in the distribution to `site.cfg` and edit the latter giving proper
directions on how to find your MKL libraries in your system.  After
doing this, you can proceed with the usual building instructions
listed above.

Pay attention to the messages during the building process in order to
know whether MKL has been detected or not.  Finally, you can check the
speed-ups on your machine by running the `bench/vml_timing.py` script
(you can play with different parameters to the
`set_vml_accuracy_mode()` and `set_vml_num_threads()` functions in the
script so as to see how it would affect performance).



.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
