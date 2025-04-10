# Train 1D regression models and visualise predictions:
juml train --model RzMlp --dataset SinMix --print_level 1 --trainer.BpSp.epochs 1000 --seed 0 --dataset.SinMix.x_std 0 --dataset.SinMix.t_std 0
juml train --model Mlp   --dataset SinMix --print_level 1 --trainer.BpSp.epochs 1000 --seed 0 --dataset.SinMix.x_std 0 --dataset.SinMix.t_std 0
juml plot1dregression --model_name dSh1i1o1te200tr200ts0.0x0.0_lM_mRZMemIex2.0m100n3pI_tBb100e1000lCle1E-05oAol0.001_s0
juml plot1dregression --model_name dSh1i1o1te200tr200ts0.0x0.0_lM_mMeIh100n3pI_tBb100e1000lCle1E-05oAol0.001_s0

# - The `juml sweep` command from `README.md` uses arguments `--Sweeper.devices
#   "[[],[],[],[],[],[]]" --Sweeper.no_cache`
# - `--Sweeper.devices "[[],[],[],[],[],[]]"` means launch 6 subprocess to run
#   sweeps, but the empty inner brackets mean assign no GPUs to any subprocess
#   (so the example works locally on any PC, even if it doesn't have GPUs)
# - To only launch a single subprocess to run all commands, do not specify
#   `--Sweeper.devices`, or equivalently specify `--Sweeper.devices "[[]]"`
#   (the default value):
juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --Sweeper.seeds 1 2 3 --Sweeper.params '{"trainer.BpSp.epochs":[100,200,300],"trainer.BpSp.optimiser.Adam.lr":[1e-5,1e-4,1e-3,1e-2]}' --Sweeper.log_x trainer.BpSp.optimiser.Adam.lr --Sweeper.no_cache

# - By default, the `juml sweep` command loads previously saved results from
#   disk when possible instead of re-running previous experiments
# - The `--Sweeper.no_cache` argument (included in the examples in `README.md`
#   and above) disables this behaviour and always runs every experiment, even
#   if results for that experiment can be found on disk
# - To enable using cached results, do not include the `--Sweeper.no_cache`
#   argument:
juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --Sweeper.seeds 1 2 3 --Sweeper.params '{"trainer.BpSp.epochs":[100,200,300],"trainer.BpSp.optimiser.Adam.lr":[1e-5,1e-4,1e-3,1e-2]}' --Sweeper.log_x trainer.BpSp.optimiser.Adam.lr

# To only sweep over Adam learning rate (and seeds) instead of both LR and
# number of epochs (and seeds), simply omit `"trainer.BpSp.epochs"` from
# `--Sweeper.params`:
juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --Sweeper.seeds 1 2 3 --Sweeper.params '{"trainer.BpSp.optimiser.Adam.lr":[1e-5,1e-4,1e-3,1e-2]}' --Sweeper.log_x trainer.BpSp.optimiser.Adam.lr --trainer.BpSp.epochs 300


# To only sweep over seeds and not hyperparameters, do not specify
# `--Sweeper.params`, or equivalently specify `--Sweeper.params "{}"` (the
# default value):
juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --Sweeper.seeds 1 2 3 --trainer.BpSp.epochs 300

# It is possible to sweep over a combination of discrete (EG model type) and
# continuous (EG learning rate and number of epochs) parameters in a single
# command:
juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --sweep.seeds 1 2 3 --sweep.params '{"trainer.BpSp.epochs":[100,200,300],"trainer.BpSp.optimiser.Adam.lr":[1e-5,1e-4,1e-3,1e-2],"model":["Mlp", "RzMlp"]}' --sweep.log_x trainer.BpSp.optimiser.Adam.lr

# To run data efficiency experiments, use `--trainer BpSpDe`, where
# `--trainer.BpSpDe.steps 600` specifies the total number of gradient steps,
# and `--trainer.BpSpDe.n_train 200` specifies the size of the subset of
# training data to use (these options can be combined with `juml sweep`)
juml train --model Mlp --model.Mlp.embedder Flatten --model.Mlp.embedder.Flatten.n 3 --dataset Mnist --trainer BpSpDe --trainer.BpSpDe.steps 600 --trainer.BpSpDe.n_train 200
