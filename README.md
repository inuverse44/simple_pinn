# PINN for 1D Wave Equation

This project implements a **Physics-Informed Neural Network (PINN)** to solve the 1D wave equation:

$$
    \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

with boundary and initial conditions:

- $ u(x, 0) = \sin(\pi x) $
- $ \frac{\partial u}{\partial t}(x, 0) = 0 $
- $ u(-1, t) = u(1, t) = 0 $

---

## 📁 Directory Structure

```
.
├── README.md
├── config.yaml
├── config_yaml_generator.py
├── requirements.txt
├── output
└── simple_PINN
    ├── __init__.py
    ├── __main__.py
    ├── postprocesses
    │   ├── __init__.py
    │   ├── evaluate_error.py
    │   ├── postprocesses.py
    │   ├── save_data.py
    │   └── visualize.py
    ├── preprocesses
    │   ├── __init__.py
    │   ├── initialize.py
    │   ├── preprocesses.py
    │   └── seed.py
    ├── settings
    │   ├── __init__.py
    │   ├── cofig_loader.py
    │   ├── config.py
    │   └── save_config.py
    ├── simple_PINN.py
    └── training
        ├── NN.py
        ├── PINN.py
        ├── __init__.py
        └── training.py
```


---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/pinn-wave.git
cd pinn-wave
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## 🚀 Run a Single or Multiple Configs

from simple_PINN.settings.config import apply_config
apply_config({ ... })
main_PINN()
▶ Batch Run from YAML
Define multiple settings in config.yaml:

```yaml
configs:
  - name: exp1
    MAX_EPOCHS_FOR_MODEL: 100
    LEARNING_RATE: 0.001
    PI_WEIGHT: 0.01
    ...
  - name: exp2
    ...
```
Run all:

python -m simple_PINN
📊 Outputs
Each run creates a unique folder in output/, e.g.:

```
output/init=100_boun=100_regi=5000_maxep=100_lr=0.001_w=0.01_v=1/
├── prediction.pdf
├── loss_history.pdf
├── difference.pdf
├── residual.pdf
├── ...
├── log.txt

```

## 🧪 Features
Supports multiple training runs via config.yaml

Logs L1/L2/max error norms

Outputs training history and prediction vs exact plots

Modular codebase with clear separation

## 📌 Requirements
Python ≥ 3.8

numpy, matplotlib, torch, scipy, PyYAML

Install with:

```bash
pip install -r requirements.txt
```

## 🧠 Author
Developed by *inverse44*.
Feel free to open issues or PRs!