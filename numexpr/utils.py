###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import logging

log = logging.getLogger(__name__)

import contextvars
import math
import os
import subprocess

from numexpr import use_vml
from numexpr.interpreter import MAX_THREADS, _get_num_threads, _set_num_threads

from . import version

if use_vml:
    from numexpr.interpreter import (_get_vml_num_threads, _get_vml_version,
                                     _set_vml_accuracy_mode,
                                     _set_vml_num_threads)


def get_vml_version():
    """
    Get the VML/MKL library version.
    """
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
        if mode not in list(acc_dict.keys()):
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
    `mkl_domain_set_num_threads(nthreads, MKL_DOMAIN_VML)` in the MKL
    library.  See:

    http://www.intel.com/software/products/mkl/docs/webhelp/support/functn_mkl_domain_set_num_threads.html

    for more info about it.
    """
    if use_vml:
        _set_vml_num_threads(nthreads)
    pass

def get_vml_num_threads():
    """
    Gets the maximum number of threads to be used in VML operations.

    This function is equivalent to the call
    `mkl_domain_get_max_threads (MKL_DOMAIN_VML)` in the MKL
    library.  See:

    http://software.intel.com/en-us/node/522118

    for more info about it.
    """
    if use_vml:
        return _get_vml_num_threads()
    return None

def set_num_threads(nthreads):
    """
    Sets a number of threads to be used in operations.

    DEPRECATED: returns the previous setting for the number of threads.

    During initialization time NumExpr sets this number to the number
    of detected cores in the system (see `detect_number_of_cores()`).
    """
    old_nthreads = _set_num_threads(nthreads)
    return old_nthreads

def get_num_threads():
    """
    Gets the number of threads currently in use for operations.
    """
    return _get_num_threads()


def _read_cgroup_cpu_limit(cgroup_dir, v2):
    """Return the CPU limit (in CPUs) set on one cgroup directory, or None."""
    try:
        if v2:
            with open(os.path.join(cgroup_dir, "cpu.max")) as f:
                quota, period = f.read().split()[:2]
            if quota == "max":
                return None
        else:
            with open(os.path.join(cgroup_dir, "cpu.cfs_quota_us")) as f:
                quota = f.read().strip()
            with open(os.path.join(cgroup_dir, "cpu.cfs_period_us")) as f:
                period = f.read().strip()
        if int(quota) <= 0 or int(period) <= 0:
            return None
        return int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit(proc_cgroup="/proc/self/cgroup", cgroup_root="/sys/fs/cgroup"):
    """Return the tightest CPU quota (in CPUs) on this process's cgroup, or None.

    Reads cgroup v2 ``cpu.max`` or cgroup v1 ``cpu.cfs_quota_us`` /
    ``cpu.cfs_period_us`` for the process's own cgroup and its ancestors, for
    example a ``docker run --cpus`` limit or a Kubernetes CPU limit. Returns
    None when no quota is set, or on non-Linux platforms where the files are
    absent.
    """
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if controllers == "":
            mounts, v2 = [cgroup_root], True
        elif "cpu" in controllers.split(","):
            names = ("cpu", "cpu,cpuacct", "cpuacct,cpu")
            mounts, v2 = [os.path.join(cgroup_root, n) for n in names], False
        else:
            continue
        for mount in mounts:
            # Walk from the process cgroup up to the mount root. Inside a
            # container without a cgroup namespace the mount root is also the
            # process cgroup, where ``path`` does not exist under the mount.
            rel = path.strip("/")
            while True:
                limit = _read_cgroup_cpu_limit(os.path.join(mount, rel), v2)
                if limit is not None:
                    limits.append(limit)
                if not rel:
                    break
                rel = os.path.dirname(rel)
    return min(limits) if limits else None


def _init_num_threads():
    """
    Detects the environment variable 'NUMEXPR_MAX_THREADS' to set the threadpool
    size, and if necessary the slightly redundant 'NUMEXPR_NUM_THREADS' or
    'OMP_NUM_THREADS' env vars to set the initial number of threads used by
    the virtual machine.
    """
    # Any platform-specific short-circuits
    if 'sparc' in version.platform_machine:
        log.warning('The number of threads have been set to 1 because problems related '
                  'to threading have been reported on some sparc machine. '
                  'The number of threads can be changed using the "set_num_threads" '
                  'function.')
        set_num_threads(1)
        return 1

    env_configured = False
    n_cores = detect_number_of_cores()
    if ('NUMEXPR_MAX_THREADS' in os.environ and os.environ['NUMEXPR_MAX_THREADS'] != '' or
        'OMP_NUM_THREADS' in os.environ and os.environ['OMP_NUM_THREADS'] != ''):
        # The user has configured NumExpr in the expected way, so suppress logs.
        env_configured = True
        n_cores = MAX_THREADS
    else:
        # The use has not set 'NUMEXPR_MAX_THREADS', so likely they have not
        # configured NumExpr as desired, so we emit info logs.
        # A cgroup CPU quota (`docker run --cpus`, a Kubernetes CPU limit, a
        # SLURM cpu bandwidth limit) caps how much CPU time the process gets
        # without restricting which cores it runs on, so it is invisible to
        # the affinity mask detect_number_of_cores() reads. Honour it too,
        # rounding up, so the default pool does not oversubscribe the quota
        # (gh-328).
        cpu_quota = _cgroup_cpu_limit()
        if cpu_quota is not None:
            n_cores = max(min(n_cores, math.ceil(cpu_quota)), 1)
        if n_cores > MAX_THREADS:
            log.info('Note: detected %d virtual cores but NumExpr set to maximum of %d, check "NUMEXPR_MAX_THREADS" environment variable.'%(n_cores, MAX_THREADS))
        if n_cores > 16:
            # Back in 2019, 8 threads would be considered safe for performance. We are in 2024 now, so adjusting.
            log.info('Note: NumExpr detected %d cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.'%n_cores)
            n_cores = 16

    # Now we check for 'NUMEXPR_NUM_THREADS' or 'OMP_NUM_THREADS' to set the
    # actual number of threads used.
    if 'NUMEXPR_NUM_THREADS' in os.environ and os.environ['NUMEXPR_NUM_THREADS'] != '':
        requested_threads = int(os.environ['NUMEXPR_NUM_THREADS'])
    elif 'OMP_NUM_THREADS' in os.environ and os.environ['OMP_NUM_THREADS'] != '':
        # Empty string is commonly used to unset the variable
        requested_threads = int(os.environ['OMP_NUM_THREADS'])
    else:
        requested_threads = n_cores
        if not env_configured:
            log.info('NumExpr defaulting to %d threads.'%n_cores)

    # The C-extension function performs its own checks against `MAX_THREADS`
    set_num_threads(requested_threads)
    return requested_threads


def detect_number_of_cores():
    """
    Detects the number of cores on a system. Cribbed from pp.

    When the process is restricted to a subset of the machine's cores -- by
    ``taskset``, ``docker run --cpuset-cpus``, or the cpuset that cluster
    managers such as SLURM pin a job to -- the CPU affinity mask reports fewer
    cores than are physically online. Prefer that mask so the default thread
    pool does not oversubscribe the cores the process may actually run on,
    which is the common failure mode when numexpr is pulled in as an indirect
    dependency (see gh-328).
    """
    # Respect a restricted CPU affinity mask where the OS exposes one (Linux).
    if hasattr(os, "sched_getaffinity"):
        try:
            ncpus = len(os.sched_getaffinity(0))
            if ncpus > 0:
                return ncpus
        except OSError:
            pass
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]))
    # Windows:
    try:
        ncpus = int(os.environ.get("NUMBER_OF_PROCESSORS", ""))
        if ncpus > 0:
            return ncpus
    except ValueError:
        pass
    return 1  # Default


def detect_number_of_threads():
    """
    DEPRECATED: use `_init_num_threads` instead.
    If this is modified, please update the note in: https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide
    """
    log.warning('Deprecated, use `init_num_threads` instead.')
    try:
        nthreads = int(os.environ.get('NUMEXPR_NUM_THREADS', ''))
    except ValueError:
        try:
            nthreads = int(os.environ.get('OMP_NUM_THREADS', ''))
        except ValueError:
            nthreads = detect_number_of_cores()

    # Check that we don't surpass the MAX_THREADS in interpreter.cpp
    if nthreads > MAX_THREADS:
        nthreads = MAX_THREADS
    return nthreads


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
            entries_to_remove = self.maxentries // 10
            for k in list(self.keys())[:entries_to_remove]:
                super(CacheDict, self).__delitem__(k)
        super(CacheDict, self).__setitem__(key, value)


class ContextDict:
    """
    A context aware version dictionary
    """
    def __init__(self):
        self._context_data = contextvars.ContextVar('context_data', default={})

    def set(self, key=None, value=None, **kwargs):
        data = self._context_data.get().copy()

        if key is not None:
            data[key] = value

        for k, v in kwargs.items():
            data[k] = v

        self._context_data.set(data)

    def get(self, key, default=None):
        data = self._context_data.get()
        return data.get(key, default)

    def delete(self, key):
        data = self._context_data.get().copy()
        if key in data:
            del data[key]
        self._context_data.set(data)

    def clear(self):
        self._context_data.set({})

    def all(self):
        return self._context_data.get()

    def update(self, *args, **kwargs):
        data = self._context_data.get().copy()

        if args:
            if len(args) > 1:
                raise TypeError(f"update() takes at most 1 positional argument ({len(args)} given)")
            other = args[0]
            if isinstance(other, dict):
                data.update(other)
            else:
                for k, v in other:
                    data[k] = v

        data.update(kwargs)
        self._context_data.set(data)

    def keys(self):
        return self._context_data.get().keys()

    def values(self):
        return self._context_data.get().values()

    def items(self):
        return self._context_data.get().items()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __delitem__(self, key):
        self.delete(key)

    def __contains__(self, key):
        return key in self._context_data.get()

    def __len__(self):
        return len(self._context_data.get())

    def __iter__(self):
        return iter(self._context_data.get())

    def __repr__(self):
        return repr(self._context_data.get())
