.. numexpr documentation master file, created by
   sphinx-quickstart on Sat Feb  4 17:19:36 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NumExpr Documentation Reference
=================================

Contents:

.. toctree::
   :maxdepth: 2

Core module
===========

.. automodule:: numexpr
   :members: evaluate, re_evaluate, disassemble, NumExpr, get_vml_version, set_vml_accuracy_mode, set_vml_num_threads, set_num_threads, detect_number_of_cores, detect_number_of_threads
   
.. py:attribute:: ncores

    The number of (virtual) cores detected.
                  
.. py:attribute:: nthreads

    The number of available threads detected.  

.. py:attribute:: version

    The version of NumExpr.      
                  
    
Tests submodule
===============

.. automodule:: numexpr.tests
   :members: test, print_versions
                  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

