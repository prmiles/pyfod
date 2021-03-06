Changelog
=========

v0.1.0 (May 8, 2019)
--------------------
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2678040.svg
   :target: https://doi.org/10.5281/zenodo.2678040
- Verified Grunwald-Letnikov implementation.
- Followed flake8, PEP8 naming conventions.
- Added user defined switch time to hybrid quadrature methods.
- Resolved #22 regarding function output of (n,) vs. (n,1) arrays.
- Allow for user defined singularity location not at upper limit.
   
v0.0.1 (April 29, 2019)
-----------------------
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2653837.svg
   :target: https://doi.org/10.5281/zenodo.2653837
- Fixed bug with hybrid quadrature methods for handling problems where lower limit does not equal 0.

v0.0.0 (April 25, 2019)
----------------------
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2651275.svg
      :target: https://doi.org/10.5281/zenodo.2651275
Initial release: Contains fractional derivative calculators for

- Riemann-Liouville
- Caputo (Pending)
- Grünwald-Letnikov (Pending)
Depending on which definition of fractional derivative you are using, you also have access to the following quadrature methods:

- Riemann Sum
- Gauss Legendre
- Gauss Laguerre
- Hybrid: Gauss Legendre, Riemann Sum
- Hybrid: Gauss Legendre, Gauss Laguerre
