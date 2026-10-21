# Motion & B0 estimation (state-by-state)

Per-shot ("state-by-state") rigid-motion and B0 estimation for EPTI
navigators. A subset of the full pipeline, carrying one path through it:
the joint motion/B0 fit solved per motion state.

This is generated from the development repository; see "Scope" below for what
is deliberately absent.

## Layout

```
matlab/motion/   motion_test_newton_bkm.m   entry point
                 motion_newton_round.m      the per-shot stage
                 motion_grad_est_spatial*   linear-gradient initialisation
                 motion_package/            vendored geometry/registration helpers
matlab/Funcs/    FFT, fitting and I/O helpers (incl. readcfl/writecfl)
python/          motion_est_lbfgs_fast.py   the solver
                 src/motion_est/            the modules it needs
```

## Setup

```bash
# python
pip install -e python/            # or put python/src on PYTHONPATH
```

```matlab
% matlab: copy the template and point it at your machine
% matlab/motion/motion_local_config.example.m -> motion_local_config.m
cfg.data_root  = '/where/your/data/lives/';
cfg.python_bin = '/path/to/env/bin/python';
```

`MOTION_DATA_ROOT` and `MOTION_PYTHON` override the config if set.

## Run

```matlab
addpath(genpath('matlab'));
motion_test_newton_bkm(dataset, series, scout, struct('st_flag',0));
```

See `matlab/motion/run_state_by_state.m` for a commented example with the
options spelled out. Defaults: fast solver, `n_alt = 2`, motion echoes
`[1 3]`, B0 echoes `[1 4 5 6]`, `fid_sel` 4 or 12 by navigator type, split
coil selection.

Echo lists are absolute echo numbers; the exported set is their union, and
every result file records what it used (`nt_sel`, `nt_mo`, `nt_dB0`).

## Scope

Included: the state-by-state fit with `motion_est_lbfgs_fast.py`.

Not included: the spatiotemporal-B0 (`st_flag = 1`) stage and its joint
solver, and the reference `motion_est_lbfgs.py`. `st_flag` should stay 0.

BART is not required — `bart()` is only called by the joint stage. The two
I/O helpers it would otherwise provide (`readcfl`, `writecfl`) ship in
`matlab/Funcs/utils`.

## Requirements

MATLAB (Signal Processing, Image Processing, Parallel Computing toolboxes)
with a GPU, and python with torch.

## Validation

Run from this tree alone, with `restoredefaultpath`, it reproduces the
development pipeline bit-for-bit (`max |diff| = 0` in `motion_est1`) on a
phantom dataset.

## Related

The continuous motion states are grouped by k-means clustering. To reconstruct
data covering all motion states in a feasible time, see the mobile-GRAPPA
method: https://github.com/linym20/Mobile-GRAPPA

Example datasets for the full demo (`example_smena_data_meGRE2.mat`, 1.2 GB;
`example_corrupted_data_meGRE2.mat`, 8.9 GB) are not tracked here:
https://drive.google.com/drive/folders/1MxaADMAppkx_47NIXhYkwR33qZwaeJKP?usp=drive_link

Note: the demo scripts those datasets drive (`motion_bkm_main.m`,
`recon_megre.m`) are not part of this estimation-only release; they remain in
the repository history at 322c8c2.
