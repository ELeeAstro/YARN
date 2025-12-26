***********************
YAML Configuration Reference
***********************

Overview
========

The retrieval configuration is specified in a YAML file (typically ``retrieval_config.yaml``)
that controls all aspects of the atmospheric retrieval: physics, opacities, parameters, and
sampling settings.

File Structure
==============

A complete configuration file has six main sections:

.. code-block:: yaml

   data:          # Observational data paths
   physics:       # Physical model choices
   opac:          # Opacity data and settings
   params:        # Retrieval parameters and priors
   sampling:      # Sampler configuration
   runtime:       # Computational settings


Data Section
============

Specifies paths to observational data and auxiliary files.

.. code-block:: yaml

   data:
     obs: ../../obs_data/planet_spectrum.txt
     stellar: ../../stellar_data/host_star.txt  # or None for brown dwarfs
     janaf: ../../JANAF_data/

**Parameters:**

- ``obs`` (string): Path to observational data file

  - Format: wavelength (µm), half-bandwidth (µm), flux, error, response_mode
  - Example: ``1.5  0.01  1.234e-14  5.6e-16  boxcar``

- ``stellar`` (string or None): Path to stellar spectrum file

  - Required for dayside emission and reflection spectroscopy
  - Set to ``None`` for brown dwarfs and directly imaged planets

- ``janaf`` (string): Path to JANAF thermochemical database

  - Required for chemical equilibrium calculations


Physics Section
================

Defines the physical model components.

.. code-block:: yaml

   physics:
     nlay: 99                        # Number of atmospheric layers

     vert_Tp: Milne_modified        # Temperature-pressure profile
     vert_alt: hypsometric          # Altitude calculation scheme
     vert_chem: constant_vmr        # Chemistry scheme
     vert_mu: dynamic               # Mean molecular weight scheme

     opac_line: lbl                 # Line opacity method (lbl or ck)
     opac_ray: lbl                  # Rayleigh scattering (lbl or None)
     opac_cia: lbl                  # CIA opacity (lbl or None)
     opac_cloud: None               # Cloud model (or None)

     rt_scheme: emission_1d         # Radiative transfer scheme
     emission_mode: brown_dwarf     # Emission mode (dayside or brown_dwarf)
     contri_func: False             # Compute contribution functions

**Atmospheric Structure:**

- ``nlay`` (integer): Number of atmospheric layers

  - Typical values: 50-100
  - More layers = higher vertical resolution but slower

**Vertical Structure Schemes:**

- ``vert_Tp`` (string): Temperature-pressure profile

  - Options: ``isothermal``, ``Barstow``, ``Milne``, ``Milne_modified``, ``Guillot``, ``MandS09``, ``picket_fence``
  - See :doc:`tp_profiles` for details

- ``vert_alt`` (string): Altitude calculation

  - Options: ``hypsometric``, ``hypsometric_variable_g``, ``hypsometric_variable_g_pref``
  - See :doc:`altitude_profiles` for details

- ``vert_chem`` (string): Chemical abundance scheme

  - Options: ``constant_vmr``, ``CE_fastchem_jax``, ``CE_rate_jax``, ``quench_approx``
  - See :doc:`chemistry` for details

- ``vert_mu`` (string): Mean molecular weight

  - Options: ``constant_mu``, ``dynamic``
  - See :doc:`mean_molecular_weight` for details

**Opacity Sources:**

- ``opac_line`` (string): Molecular line opacities

  - Options: ``lbl`` (line-by-line), ``ck`` (correlated-k), ``None``
  - See :doc:`opacity_line` and :doc:`opacity_ck` for details

- ``opac_ray`` (string or None): Rayleigh scattering

  - Options: ``lbl``, ``None``
  - See :doc:`opacity_rayleigh` for details

- ``opac_cia`` (string or None): Collision-induced absorption

  - Options: ``lbl``, ``None``
  - See :doc:`opacity_cia` for details

- ``opac_cloud`` (string or None): Cloud opacities

  - Options: ``grey_cloud``, ``powerlaw_cloud``, ``F18_cloud``, ``direct_nk``, ``None``
  - See :doc:`opacity_clouds` for details

**Radiative Transfer:**

- ``rt_scheme`` (string): RT geometry

  - Options: ``transmission_1d``, ``emission_1d``, ``albedo_1d``
  - See :doc:`radiative_transfer` for details

- ``emission_mode`` (string): Emission geometry (only for ``emission_1d``)

  - Options: ``dayside`` (secondary eclipse), ``brown_dwarf`` (self-luminous)

- ``contri_func`` (boolean): Compute contribution functions

  - ``True``: Calculate contribution functions (slower)
  - ``False``: Skip contribution functions (faster)


Opac Section
============

Specifies opacity data files and settings.

.. code-block:: yaml

   opac:
     wl_master: wl_dnu_1.txt       # Master wavelength grid file
     full_grid: False              # Use full or cut wavelength grid
     ck: False                     # Use correlated-k tables
     ck_mix: RORR                  # c-k mixing scheme (RORR or PRAS)

     line:                         # Line opacity files
       - {species: H2O, path: ../../opac_data/lbl/H2O_dnu_1.npz}
       - {species: CO, path: ../../opac_data/lbl/CO_dnu_1.npz}

     ray:                          # Rayleigh scattering species
       - {species: H2}
       - {species: He}

     cia:                          # CIA opacity files
       - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
       - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}

     cloud: None                   # Cloud opacity files (or None)

**General Settings:**

- ``wl_master`` (string): Filename of master wavelength grid

  - Must exist in ``opac_data/`` directory
  - Defines high-resolution wavelength sampling

- ``full_grid`` (boolean): Use full or cut wavelength grid

  - ``True``: Use entire master grid (slower, more memory)
  - ``False``: Cut to observed wavelength range (faster, recommended)

- ``ck`` (boolean): Enable correlated-k opacities

  - ``True``: Use c-k tables (faster)
  - ``False``: Use line-by-line (slower, more accurate)

- ``ck_mix`` (string): c-k mixing scheme (only if ``ck: True``)

  - Options: ``RORR`` (Random Overlap with Resort and Rebin), ``PRAS`` (Pre-mixed)

**Opacity File Lists:**

Each opacity source (``line``, ``ray``, ``cia``, ``cloud``) is specified as a list of dictionaries:

.. code-block:: yaml

   line:
     - {species: H2O, path: ../../opac_data/lbl/H2O_dnu_1.npz}
     - {species: CO, path: ../../opac_data/lbl/CO_dnu_1.npz}

- ``species`` (string): Chemical species identifier
- ``path`` (string): Relative or absolute path to opacity file


Params Section
==============

Defines all retrieval parameters, priors, and transformations.

Parameter Format
----------------

Each parameter is specified as a dictionary:

.. code-block:: yaml

   params:
     - { name: log_10_g, dist: uniform, low: 4.0, high: 5.5, transform: logit, init: 5.0 }
     - { name: T_int, dist: uniform, low: 500.0, high: 1500.0, transform: logit, init: 1000.0 }
     - { name: R_p, dist: delta, value: 1.0, transform: identity, init: 1.0 }

**Required Fields:**

- ``name`` (string): Parameter name

  - Must match parameter names expected by chosen physics schemes

- ``dist`` (string): Prior distribution type

  - ``uniform``: Uniform prior (requires ``low``, ``high``)
  - ``gaussian`` or ``normal``: Gaussian prior (requires ``mu``, ``sigma``)
  - ``lognormal``: Log-normal prior (requires ``mu``, ``sigma``)
  - ``delta``: Fixed value (requires ``value``)

- ``transform`` (string): Parameter transformation

  - ``identity``: No transformation
  - ``logit``: Logit transformation (maps [low, high] to [-∞, +∞])
  - ``log``: Logarithmic transformation

- ``init`` (float): Initial/test value for forward model testing

**Distribution-Specific Fields:**

For ``uniform`` distribution:

.. code-block:: yaml

   - { name: T_int, dist: uniform, low: 500.0, high: 1500.0, transform: logit, init: 1000.0 }

- ``low`` (float): Lower bound
- ``high`` (float): Upper bound

For ``gaussian``/``normal`` distribution:

.. code-block:: yaml

   - { name: log_10_g, dist: gaussian, mu: 5.0, sigma: 0.5, transform: identity, init: 5.0 }

- ``mu`` (float): Mean
- ``sigma`` (float): Standard deviation

For ``delta`` (fixed) distribution:

.. code-block:: yaml

   - { name: R_s, dist: delta, value: 0.0, transform: identity, init: 0.0 }

- ``value`` (float): Fixed value

Parameter Categories
--------------------

**Planetary Parameters:**

.. code-block:: yaml

   - { name: R_p, dist: uniform, low: 0.9, high: 1.2, transform: logit, init: 1.0 }
   - { name: M_p, dist: uniform, low: 1.0, high: 80.0, transform: logit, init: 20.0 }
   - { name: log_10_g, dist: uniform, low: 2.0, high: 5.5, transform: logit, init: 3.0 }

- ``R_p``: Planetary radius (R_Jupiter)
- ``M_p``: Planetary mass (M_Jupiter)
- ``log_10_g``: Log₁₀ surface gravity (cm s⁻²)

**Observational Parameters:**

.. code-block:: yaml

   - { name: R_s, dist: delta, value: 1.0, transform: identity, init: 1.0 }
   - { name: D, dist: delta, value: 10.0, transform: identity, init: 10.0 }

- ``R_s``: Stellar radius (R_Sun) - set to 0 for brown dwarfs
- ``D``: Distance to object (pc) - for brown dwarfs/directly imaged planets

**Atmospheric Boundaries:**

.. code-block:: yaml

   - { name: p_bot, dist: delta, value: 1000.0, transform: identity, init: 1000.0 }
   - { name: p_top, dist: delta, value: 1e-4, transform: identity, init: 1e-4 }
   - { name: log_10_p_ref, dist: delta, value: 0.0, transform: identity, init: 0.0 }

- ``p_bot``: Bottom pressure (bar)
- ``p_top``: Top pressure (bar)
- ``log_10_p_ref``: Log₁₀ reference pressure (bar)

**Chemistry Parameters:**

.. code-block:: yaml

   - { name: log_10_f_H2O, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }
   - { name: log_10_f_CO, dist: uniform, low: -9, high: -1, transform: logit, init: -5 }
   - { name: log_10_f_CH4, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }

- ``log_10_f_X``: Log₁₀ volume mixing ratio of species X

**Noise Parameters:**

.. code-block:: yaml

   - { name: c, dist: uniform, low: -14.0, high: -7.0, transform: logit, init: -10.0 }

- ``c``: Log₁₀ jitter/systematic error scaling


Sampling Section
================

Configures the nested sampling or MCMC algorithm.

.. code-block:: yaml

   sampling:
     engine: jaxns                 # Sampler choice (jaxns, nuts, blackjax_ns, ultranest, dynesty)

     nuts:                         # NUTS/MCMC settings (if engine: nuts)
       backend: numpyro
       warmup: 1000
       draws: 1000
       seed: 42
       chains: 4

     jaxns:                        # JAXNS settings (if engine: jaxns)
       max_samples: 100000
       num_live_points: 500
       s: 4
       k: 0
       c: null
       shell_fraction: 0.5
       difficult_model: false
       parameter_estimation: true
       gradient_guided: false
       init_efficiency_threshold: 0.1
       verbose: true

       termination:
         ess: 10000
         evidence_uncert: null
         dlogZ: 1.0
         max_samples: 100000

       posterior_samples: 10000
       seed: 42

     blackjax_ns:                  # BlackJAX NS settings
       num_live_points: 150
       num_inner_steps: 50
       num_delete: 75
       dlogz_stop: 0.01
       seed: 42

     ultranest:                    # UltraNest settings
       num_live_points: 500
       min_num_live_points: 500
       dlogz: 0.5
       max_iters: 0
       verbose: true
       show_status: true

     dynesty:                      # Dynesty settings
       nlive: 500
       bound: multi
       sample: auto
       dlogz: 1.0

**Engine Selection:**

- ``engine`` (string): Sampling algorithm

  - ``jaxns``: Nested sampling with JAXNS (recommended for evidence)
  - ``nuts``: NUTS/HMC with NumPyro (recommended for posterior only)
  - ``blackjax_ns``: Nested sampling with BlackJAX
  - ``ultranest``: Nested sampling with UltraNest
  - ``dynesty``: Nested sampling with Dynesty

**JAXNS Settings:**

Core Configuration:

- ``max_samples`` (integer): Hard cap on nested sampling iterations (10000-100000)
- ``num_live_points`` (integer): Number of live points (100-1000)

  - More points = better posterior resolution but slower
  - Typical: 500 for parameter estimation

Slice Sampler:

- ``s`` (integer): Slices per dimension (3-5 typical)
- ``k`` (integer): Phantom samples (0 for simple models, >0 for multimodal)
- ``c`` (integer or null): Parallel chains (null = auto)
- ``shell_fraction`` (float): Fraction of shell explored per step (0.5 typical)

Behavior Flags:

- ``difficult_model`` (boolean): Use robust defaults for challenging posteriors
- ``parameter_estimation`` (boolean): Optimize for posterior quality (not just evidence)
- ``gradient_guided`` (boolean): Use gradients for proposals (experimental)
- ``init_efficiency_threshold`` (float): Initial uniform sampling efficiency threshold
- ``verbose`` (boolean): Print progress to terminal

Termination Criteria:

- ``ess`` (integer): Target effective sample size (5000-20000)
- ``evidence_uncert`` (float or null): Stop when σ(log Z) < threshold
- ``dlogZ`` (float): Stop when remaining evidence contribution < threshold
- ``max_samples`` (integer): Safety cap for iterations

Output:

- ``posterior_samples`` (integer): Number of equal-weight posterior samples (5000-10000)
- ``seed`` (integer): Random seed for reproducibility

**NUTS Settings:**

- ``backend`` (string): MCMC backend (``numpyro`` recommended)
- ``warmup`` (integer): Number of warmup iterations (500-2000)
- ``draws`` (integer): Number of posterior samples per chain (500-2000)
- ``seed`` (integer): Random seed
- ``chains`` (integer): Number of parallel MCMC chains (4 typical)


Runtime Section
===============

Controls computational resources and JAX configuration.

.. code-block:: yaml

   runtime:
     platform: gpu                 # Compute platform (cpu, gpu, cuda, metal)
     cuda_visible_devices: 0       # GPU device selection (0, "0,1", etc.)
     threads: 1                    # CPU thread count

**Parameters:**

- ``platform`` (string): Compute device

  - ``cpu``: CPU only
  - ``gpu``: Auto-detect GPU (CUDA or Metal)
  - ``cuda``: Force CUDA (NVIDIA)
  - ``metal``: Force Metal (Apple Silicon)

- ``cuda_visible_devices`` (integer or string): GPU device selection

  - Single GPU: ``0``, ``1``, etc.
  - Multiple GPUs: ``"0,1"``, ``"0,1,2,3"``
  - Only used if platform is GPU/CUDA

- ``threads`` (integer): Number of CPU threads

  - For CPU-only runs
  - Typical: 1-8 depending on available cores


Complete Example
================

A full retrieval configuration for a brown dwarf:

.. code-block:: yaml

   data:
     obs: ../../obs_data/Gliese_229B_emission.txt
     stellar: None
     janaf: ../../JANAF_data/

   physics:
     nlay: 99
     vert_Tp: Milne_modified
     vert_alt: hypsometric
     vert_chem: constant_vmr
     vert_mu: dynamic
     opac_line: lbl
     opac_ray: None
     opac_cia: lbl
     opac_cloud: None
     rt_scheme: emission_1d
     emission_mode: brown_dwarf
     contri_func: False

   opac:
     wl_master: wl_dnu_1.txt
     full_grid: False
     ck: False
     ck_mix: RORR

     line:
       - {species: H2O, path: ../../opac_data/lbl/H2O_dnu_1.npz}
       - {species: CO, path: ../../opac_data/lbl/CO_dnu_1.npz}
       - {species: CH4, path: ../../opac_data/lbl/CH4_dnu_1.npz}

     ray: None

     cia:
       - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
       - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}

     cloud: None

   params:
     - { name: R_s, dist: delta, value: 0.0, transform: identity, init: 0.0 }
     - { name: D, dist: delta, value: 10.0, transform: identity, init: 10.0 }
     - { name: p_bot, dist: delta, value: 1000.0, transform: identity, init: 1000.0 }
     - { name: p_top, dist: delta, value: 1e-4, transform: identity, init: 1e-4 }
     - { name: log_10_p_ref, dist: delta, value: 0.0, transform: identity, init: 0.0 }

     - { name: log_10_g, dist: uniform, low: 4.0, high: 5.5, transform: logit, init: 5.0 }
     - { name: R_p, dist: uniform, low: 0.5, high: 2.0, transform: logit, init: 1.0 }

     - { name: T_int, dist: uniform, low: 500.0, high: 1500.0, transform: logit, init: 1000.0 }
     - { name: T_skin, dist: uniform, low: 100.0, high: 500.0, transform: logit, init: 300.0 }
     - { name: log_10_k_ir, dist: uniform, low: -6, high: 6, transform: logit, init: -2 }
     - { name: log_10_p_0, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: beta, dist: uniform, low: 0.3, high: 1.0, transform: logit, init: 0.55 }

     - { name: log_10_f_H2O, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }
     - { name: log_10_f_CO, dist: uniform, low: -9, high: -1, transform: logit, init: -5 }
     - { name: log_10_f_CH4, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }

     - { name: c, dist: uniform, low: -14.0, high: -7.0, transform: logit, init: -10.0 }

   sampling:
     engine: jaxns

     jaxns:
       max_samples: 100000
       num_live_points: 500
       s: 4
       k: 0
       c: null
       shell_fraction: 0.5
       difficult_model: false
       parameter_estimation: true
       gradient_guided: false
       init_efficiency_threshold: 0.1
       verbose: true

       termination:
         ess: 10000
         evidence_uncert: null
         dlogZ: 1.0
         max_samples: 100000

       posterior_samples: 10000
       seed: 42

   runtime:
     platform: gpu
     cuda_visible_devices: 0
     threads: 1


See Also
========

- :doc:`tp_profiles` for T-p profile parameters
- :doc:`chemistry` for chemistry scheme parameters
- :doc:`opacity_line` for opacity file formats
- :doc:`radiative_transfer` for RT scheme options
