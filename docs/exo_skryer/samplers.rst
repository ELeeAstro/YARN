********
Samplers
********

Exo_Skryer supports several sampling backends. The sampler is selected with
``sampling.engine`` in the YAML configuration.

Available engines
=================

- ``jaxns``: Nested sampling via JAXNS (evidence-focused, JAX-native).
- ``blackjax_ns``: Nested sampling via BlackJAX.
- ``ultranest``: Nested sampling via UltraNest (pip-installable replacement for MultiNest).
- ``dynesty``: Nested sampling via Dynesty.
- ``nuts``: HMC/NUTS MCMC (NumPyro or BlackJAX backend).

UltraNest
=========

UltraNest is the replacement for MultiNest-based workflows. It is
pure-Python and pip-installable:

.. code-block:: bash

   pip install ultranest

Minimal YAML block:

.. code-block:: yaml

   sampling:
     engine: ultranest
     ultranest:
       num_live_points: 500
       min_num_live_points: 500
       dlogz: 0.5
       max_iters: 0
       verbose: true
       show_status: true

Dynesty
=======

.. code-block:: yaml

   sampling:
     engine: dynesty
     dynesty:
       nlive: 500
       bound: multi
       sample: auto
       dlogz: 1.0

JAXNS
=====

.. code-block:: yaml

   sampling:
     engine: jaxns
     jaxns:
       max_samples: 100000
       num_live_points: 500
       s: 4
       k: 0
       shell_fraction: 0.5
       parameter_estimation: true
       verbose: true
       termination:
         dlogZ: 1.0
       posterior_samples: 10000

BlackJAX nested sampling
========================

.. code-block:: yaml

   sampling:
     engine: blackjax_ns
     blackjax_ns:
       num_live_points: 150
       num_inner_steps: 50
       num_delete: 75
       dlogz_stop: 0.01

NUTS / HMC
==========

.. code-block:: yaml

   sampling:
     engine: nuts
     nuts:
       backend: numpyro
       warmup: 1000
       draws: 1000
       seed: 42
       chains: 4
