# Beam Control

## Environment setup
### Requirements
- Miniconda
- Python 3.8

### Create Environment
    $ conda env create -f environment.yml

### Enable Environment
    $ conda activate fmls-beam-control

### Fetch training dataset
- Download ```data release.csv``` from [BOOSTR: A Dataset for Accelerator Control Systems (Partial Release 2020)](https://zenodo.org/record/4088982#.YhAB-ZPMJAc)
    - Rename this file to ```data_release.csv```
- Alternatively, download the data files for each of the following signals: ```['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']``` from [BOOSTR: A Dataset for Accelerator Control Systems (Full Release 2020)](https://zenodo.org/record/4382663#.YvFGouzMJQI)
    - Combine these datafiles into one ```data_release.csv``` file, with each
      variable corresponding to a different column
- Create a `data/` subdirectory and place ```data_release.csv``` inside

## Surrogate Model Training
    $ python train_surrogate.py

- Training results will appear in `results/plots/surrogate_plots/`
- Update `LATEST_SURROGATE_MODEL` in `globals.py` with the path to the new model
  for agent training
- You may configure ```globals.py``` to modify surrogate training
    - Refer to the [Appendix](#Appendix) for more details

## Agent Training
    $ python run_training.py

- Update `LATEST_AGENT_MODEL` in `globals.py` with new model for play mode
- You may configure ```globals.py``` to modify agent training
    - Set ```ARCH_TYPE``` to ```MLP_Quantized``` to train the quantized agent
    - Refer to the [Appendix](#Appendix) for more details

## Appendix:

The following variables in the `globals.py` file are used to homogenize the training spanning across different files but using shared variables

| Variable               | Purpose                                                                                                                                    |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| DATA_CONFIG            | Stores the address to the `.json` file that is used to pull the data for training/ testing.                                                |
| LOOK_BACK              | Number of ticks in the look back cycle 1 second (cycle - 15Hz).                                                                            |
| LOOK_FORWARD           | Number of ticks in the look forward cycle                                                                                                  |
| VARIABLES              | Top causal variables influencing the outputs.                                                                                              |
| OUTPUTS                | Variables to be considered as outputs during training. MUST include `B:VIMIN`                                                              |
| NSTEPS                 | Data entries to be used for  training/ testing.                                                                                            |
| N_SPLITS               | Number of k-fold validation splits to train the surrogate.                                                                                 |
| EPOCHS                 | Epochs for training the surrogate                                                                                                          |
| BATCHES                | Batch size for surrogate training                                                                                                          |
| SURROGATE_VERSION      | Version number for the current surrogate.                                                                                                  |
| CKPT_FREQ              | checkpoint frequency                                                                                                                       |
| SURROGATE_CKPT_DIR     | subfolder to store model checkpoints                                                                                                       |
| SURROGATE_DIR          | subfolder to store final trained surrogate                                                                                                 |
| SURROGATE_FILE_NAME    | Common prefix to append to surrogate file name.                                                                                            |
| DQN_CONFIG_FILE        | `.json` file that store training hyperparameters for the agent.                                                                            |
| ARCH_TYPE              | Type of model architecture. Can be 'MLP', 'MLP_Ensemble', or 'LSTM'.                                                                       |
| NMODELS                | Number of models in the architecture. Useful when creating a  'MLP_Ensemble'. Default set to 1 otherwise.                                  |
| LATEST_SURROGATE_MODEL | Absolute path address to the latest surrogate model.                                                                                       |
| ENV_TYPE               | Can be "discrete" or "continuous"                                                                                                          |
| ENV_VERSION            | Version of accelerator env to be used for agent training.                                                                                  |
| AGENT_EPISODES         | Training epochs for the policy model.                                                                                                      |
| AGENT_NSTEPS           | Steps per episode for the policy model.                                                                                                    |
| IN_PLAY_MODE           | If `False` the code run is for training the agent, otherwise the agent is being used in test mode                                          |
| CORR_PLOTS_DIR         | Directory to save the correlation plot for the predictions made using RL agent on the data.                                                |
| EPISODES_PLOTS_DIR     | Directory to save the `B:VIMIN` and `B:IMINER` as predicted by the RL agent.                                                               |
| DQN_SAVE_DIR           | Directory to save the final and best_episode models for the agent.                                                                         |
| LATEST_AGENT_MODEL     | Absolute path to the policy model to be used in the PLAY MODE.                                                                             |
