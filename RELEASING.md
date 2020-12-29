Releasing NumExpr
==================

* Author: Robert A. McLeod
* Contact: robbmcleod@gmail.com
* Date: 2020-12-28

Following are notes for releasing NumExpr.

Preliminaries
-------------

* Make sure that `RELEASE_NOTES.rst` and `ANNOUNCE.rst` are up to date with the latest news in the release.
* Remove the `.devN` suffix in `numexpr/version.py`.
* Do a commit and a push:

    `git commit -a -m "Getting ready for release X.Y.Z"`

* If the directories `dist` or `artifact` exist delete them.

Local Testing
-------------

* Re-compile locally with MKL support and see if all tests passes as well.
* Run all the benchmarks in `bench/` directory and see if the
  speed-ups are the expected ones.

Tagging
-------

* Create a tag `vX.Y.Z` from `master` and push the tag to GitHub:

    `git tag -a vX.Y.Z -m "Tagging version X.Y.Z"`
    `git push`
    `git push --tags`

* If you happen to have to delete the tag, such as artifacts demonstrates a fault, first delete it locally,

    `git tag --delete vX.Y.Z`

  and then remotely on Github,

    `git push --delete origin vX.Y.Z`

Build Wheels
------------

* Check on GitHub Actions `github.com/robbmcleod/cpufeature/actions` that all the wheels built successfully.
* Download `artifacts.zip` and unzip.
* Make the source tarball with the command

    `python setup.py sdist`

Releasing
---------

* Upload the built wheels to PyPi via Twine.

    `twine upload artifact/numexpr*.whl`

* Upload the source distribution.

    `twine upload dist/numexpr-X.Y.Z.tar.gz`

* Check on `pypi.org/project/numexpr/#files` that the wheels and source have uploaded as expected.

Announcing
----------

* Send an announcement to the NumPy list, PyData and python-announce
  list.  Use the `ANNOUNCE.rst` file as skeleton (or possibly as the
  definitive version). Email should be addressed to the following lists:
  * python-announce-list@python.org
  * numpy-discussion@python.org
  * pydata@googlegroups.com

Post-release actions
--------------------

* Edit `numexpr/version.py` to bump the version revision
  (i.e. X.Y.Z --> X.Y.(Z+1).dev0).
* Create new headers for adding new features in `RELEASE_NOTES.rst`
  and add this place-holder:

  `* **Under development.**`

  Don't forget to update header to the next version in those files.

* Commit your changes:

  `git commit -a -m "Post X.Y.Z release actions done"`
  `git push`

Fin.
