from numexpr import use_vml

if use_vml:
    from numexpr.interpreter import (
        _get_vml_version, _set_vml_accuracy_mode, _set_vml_num_threads)


def get_vml_version():
    """Get the VML/MKL library version."""
    if use_vml:
        return _get_vml_version()
    else:
        return None


def set_vml_accuracy_mode(mode):
    """
    Set the accuracy mode for VML operations.

    The `mode` parameter can take the values:
    - 'high': high accuracy mode (HA), <1 least significant bit
    - 'low': low accuracy mode (LA), typically 1-2 least significant bits
    - 'fast': enhanced performance mode (EP)
    - None: mode settings are ignored

    This call is equivalent to the `vmlSetMode()` in the VML library.
    See:

    http://www.intel.com/software/products/mkl/docs/webhelp/vml/vml_DataTypesAccuracyModes.html

    for more info on the accuracy modes.

    Returns old accuracy settings.
    """
    if use_vml:
        acc_dict = {None: 0, 'low': 1, 'high': 2, 'fast': 3}
        acc_reverse_dict = {1: 'low', 2: 'high', 3: 'fast'}
        if mode not in acc_dict.keys():
            raise ValueError(
                "mode argument must be one of: None, 'high', 'low', 'fast'")
        retval = _set_vml_accuracy_mode(acc_dict.get(mode, 0))
        return acc_reverse_dict.get(retval)
    else:
        return None


def set_vml_num_threads(nthreads):
    """
    Suggests a maximum number of threads to be used in VML operations.

    This function is equivalent to the call
    `mkl_domain_set_num_threads(nthreads, MKL_VML)` in the MKL
    library.  See:

    http://www.intel.com/software/products/mkl/docs/webhelp/support/functn_mkl_domain_set_num_threads.html

    for more info about it.
    """
    if use_vml:
        _set_vml_num_threads(nthreads)


class CacheDict(dict):
    """
    A dictionary that prevents itself from growing too much.
    """

    def __init__(self, maxentries):
        self.maxentries = maxentries
        super(CacheDict, self).__init__(self)

    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in self.keys()[:entries_to_remove]:
                super(CacheDict, self).__delitem__(k)
        super(CacheDict, self).__setitem__(key, value)

