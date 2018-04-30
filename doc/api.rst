NumExpr API
===========

.. automodule:: numexpr
   :members: evaluate, re_evaluate, disassemble, NumExpr, get_vml_version, set_vml_accuracy_mode, set_vml_num_threads, set_num_threads, detect_number_of_cores, detect_number_of_threads
   
.. py:attribute:: ncores

    The number of (virtual) cores detected.
                  
.. py:attribute:: nthreads

    The number of threads currently in-use.

.. py:attribute:: MAX_THREADS

    The maximum number of threads, as set by the environment variable ``NUMEXPR_MAX_THREADS``

.. py:attribute:: version

    The version of NumExpr.      
                  
    
Tests submodule
---------------

.. automodule:: numexpr.tests
   :members: test, print_versions