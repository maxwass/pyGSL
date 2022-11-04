******************************************************************
pyGSL: A Scalable Library for Graph Structure Learning
******************************************************************

.. image:: https://img.shields.io/badge/license-MIT-red
    :target: https://opensource.org/licenses/MIT
    :alt: Licensed under the MIT license
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
    :target: https://docs.python.org/3.9/
    :alt: Python Version
.. image:: https://img.shields.io/badge/version%20control-git-blue.svg?logo=github
    :target: https://github.com/SalishSeaCast/rpn-to-gemlam
    :alt: Git on GitHub
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://black.readthedocs.io/en/stable/
    :alt: The uncompromising Python code formatter
.. image:: https://readthedocs.org/projects/rpn-to-gemlam/badge/?version=latest
    :target: https://rpn-to-gemlam.readthedocs.io/en/latest/
    :alt: Documentation Status
.. image:: https://img.shields.io/github/issues/SalishSeaCast/rpn-to-gemlam?logo=github
    :target: https://github.com/SalishSeaCast/rpn-to-gemlam/issues
    :alt: Issue Tracker

``pyGSL`` houses state-of-the-art implementations of graph structure learning (also called 'network topology inference'
or simply 'graph learning') models, as well as synthetic and real datasets across a variety of domains.

``pyGSL`` houses 4 types of models: ad-hoc, model-based, unrolling-based, and deep-learning-based. Model-based formulations often admit
iterative solution methods, and are implemented in GPU friendly ways when feasible. The unrolling-based methods
leverage the concept of `algorithm unrolling`_ to learn the values of the optimization parameters in the model-based
methods using a dataset. We build such models using `Pytorch-Lightning`_ making it easy
to scale models to (multi-)GPU training environments when needed.

Synthetic datasets include a wide range of network classes and many signal constructions (e.g. smooth, diffusion, etc).
Real datasets include neuroimaging data (HCP-YA structural/functional connectivity graphs), social network co-location data, and more.

Installation & Setup
====
See instructions in ``/envs/environment.yml`` for instructions on how to setup the required conda environment. Only tested on macOS and Ubuntu systems.

License
======

Please observe the MIT license that is listed in this repository.

.. _Pytorch-Lightning: https://www.pytorchlightning.ai
.. _algorithm unrolling: https://arxiv.org/abs/1912.10557
.. followed: https://ubc-moad-docs.readthedocs.io/en/latest/python_packaging/pkg_structure.html
