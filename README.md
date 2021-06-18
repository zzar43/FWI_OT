# Full waveform inversion with unbalanced optimal transport distance and a mixed Wasserstein distance

Preprint paper: Application of an unbalanced optimal transport distance and a mixed L1/Wasserstein distance to full waveform inversion [arxiv link](https://arxiv.org/abs/2004.05237)

## Installation
```
git clone https://github.com/zzar43/FWI_OT
```

## How to use

A Marmousi test data is saved in the folder ``marmousi_model''.

Forward modelling with the true model (parallel computing with 2 workers):
```
julia -p 2 make_data.jl
```
The received signal generated with true model are saved in the folder ``temp_data''.

Perform full waveform inversion with the L2 objective function:
```
julia -p 2 inversion_l2.jl
```

Perform full waveform inversion with the mixed Wasserstein distance objective function:

```
julia -p 2 inversion_mixed_ot.jl
```

Perform full waveform inversion with the unbalanced optimal transport distance objective function:

```
julia -p 2 inversion_uot.jl
```

The inverse results for each iterations are saved in the folder ``temp_data''.

## Demo

Here is a demo:
![Marmousi model]
(https://github.com/zzar43/FWI_OT/demo/marmousi.jpg)