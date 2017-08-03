#!/bin/bash
# http://svn.python.org/projects/python/trunk/Misc/README.valgrind
# http://stackoverflow.com/questions/3982036/how-can-i-use-valgrind-with-python-c-extensions
valgrind --tool=memcheck --leak-check=full -v --suppressions=valgrind-python.supp \
                                          python -E -tt numexpr3/tests/simple_ne3.py
