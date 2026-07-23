# Motion-corrected multi-echo GRE reconstruction

Two-step pipeline: estimate per-shot motion + B0 from SMENA navigators, then use
that estimate to reconstruct a motion-corrected multi-echo GRE image.

1. **Motion estimation** — run *either*:
   - `motion_bkm_main.m` (MATLAB, quasi-Newton `fminunc` fit), or
   - `motion_bkm_python.py` (pure Python, LBFGS fit — loads the raw `.mat` directly,
     no MATLAB round-trip; substantially faster on GPU)

   Both read `data/example_smena_data_meGRE2.mat` and save the same output file,
   `data/motion_estimation.mat` (`motion_est`, `cr_est`, `Bas`, `Bas_norm`,
   `voxel_size`, `size_nav`, `fid_sel`, `fid_dis`, `nshot_total`), so either one
   can feed step 2.

2. **Reconstruction** — run `recon_megre.m`. It loads `data/motion_estimation.mat`
   (from step 1) plus `data/example_corrupted_data_meGRE2.mat` (the raw multi-echo
   k-space) and reconstructs motion-corrected + uncorrected images for a few
   representative echoes.

   in this code, the continuous motion states are divided into multiple groups by kmeans clustering. To allow the reconstion of data including all motion states within a feasible time, please check out mobile-GRAPPA method: https://github.com/linym20/Mobile-GRAPPA.

## Layout

```
motion_bkm_main.m       step 1, MATLAB
motion_bkm_python.py    step 1, Python
recon_megre.m           step 2

matlab/
  Funcs/                 reconstruction + motion-operator helpers (own code)
  fft/                    small FFT helpers
  motion/                 core motion-estimation pipeline (own code)

python/
  motion_est-main/        motion_bkm_python.py's estimation package + environment.yml

data/
  example_smena_data_meGRE2.mat        SMENA navigator data (input to step 1)
  example_corrupted_data_meGRE2.mat    raw multi-echo GRE k-space (input to step 2)
  motion_estimation.mat                 step 1's output (input to step 2) — not
                                         included; generated when you run step 1
```

## Getting the example data

`data/*.mat` are not tracked in this repo (`example_smena_data_meGRE2.mat` is 1.2GB,
`example_corrupted_data_meGRE2.mat` is 8.9GB — both well over GitHub's size limits).
Download them here and place them under `data/`: 
https://drive.google.com/drive/folders/1MxaADMAppkx_47NIXhYkwR33qZwaeJKP?usp=drive_link

## Setting up MATLAB

Both `motion_bkm_main.m` and `recon_megre.m` call `addpath(genpath('./matlab'))`
themselves, so no manual path setup is needed — just `cd` into this folder before
running either one.

## Setting up Python (for `motion_bkm_python.py`)

```bash
conda env create -f python/motion_est-main/environment.yml
conda activate motion_est
python motion_bkm_python.py
```

This creates a `motion_est` conda env with `torch`/`cupy`/`triton` pinned to CUDA
11.8 — a GPU is required. `motion_bkm_python.py` defaults to
`data/example_smena_data_meGRE2.mat` and auto-picks the GPU with the most free
memory; see `python motion_bkm_python.py --help` for overrides (`--mat_path`,
`--gpu`, `--out`, etc).

The `python/motion_est-main/src/motion_est` package started as a fork of Daniel
Abraham's [`mr_recon`](https://github.com/danielabrahamgit/mr_recon) and has since
been extended for this pipeline.

## Notes

- `recon_megre.m` assumes an NVIDIA GPU (`useGPU = 1`, `gpuArray`/`gather` calls —
  Parallel Computing Toolbox) and uses MATLAB's `kmeans` (Statistics and Machine
  Learning Toolbox) and `imresize3` (Image Processing Toolbox).
- `matlab/motion/motion_test_newton_bkm.m` is a separate, older exploratory script
  also included in this bundle (not part of the step 1 → step 2 pipeline above).
  It's not currently runnable as-is — it depends on BART and on a `external/yannick/`
  registration toolbox that isn't included here — kept for reference only.
