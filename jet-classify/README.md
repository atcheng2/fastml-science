# Jet Classification

## Environment setup
### Requirements
- Miniconda
- Python 3.8

### Create Environment
    $ conda env create -f environment.yml

### Enable Environment
    $ conda activate fmls-jet-classify

## Run Training
    $ python train.py -c <float_baseline|quantized_baseline>.yml

## Sample Results

### Training Float Baseline:

```
python3 train.py -c float_baseline.yml
```
![Alt text](model/float_baseline/keras_roc_curve.png?raw=true "Float Baseline ROC Curve")

`Model test accuracy = 0.766`

`Model test weighted average AUC = 0.943`

### Training Quantized Baseline:

```
python3 train.py -c quantized_baseline.yml
```
![Alt text](model/quantized_baseline/keras_roc_curve.png?raw=true "Quantized Baseline ROC Curve")

`Model test accuracy = 0.764`

`Model test weighted average AUC = 0.941`
