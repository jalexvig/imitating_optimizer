# Imitating Optimizers

See [this post](http://jalexvig.github.io/blog/imitating-an-optimizer/) for more details.

### Example use

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --optimizer_model sgd --optimizer_kwargs_model lr:0.01 --num_steps_meta 10000 --seed 1 --num_steps_model 5 --freq_debug 100
```

### Options

```
usage: main.py [-h] [--model {binary,mnist}] [--model_dir MODEL_DIR]
               [--seed SEED] [--optimizer_model {sgd,adadelta,adam}]
               [--optimizer_meta {sgd,adadelta,adam}]
               [--optimizer_kwargs_model OPTIMIZER_KWARGS_MODEL]
               [--optimizer_kwargs_meta OPTIMIZER_KWARGS_META]
               [--num_steps_model NUM_STEPS_MODEL]
               [--num_steps_meta NUM_STEPS_META] [--reset] [--supply_truth]
               [--run_name RUN_NAME] [--test TEST] [--freq_save FREQ_SAVE]
               [--freq_debug FREQ_DEBUG]

optional arguments:
  -h, --help            show this help message and exit
  --model {binary,mnist}
                        Name of model.
  --model_dir MODEL_DIR
                        Directory to save run information in.
  --seed SEED           Random seed.
  --optimizer_model {sgd,adadelta,adam}
                        Number of steps to run model
  --optimizer_meta {sgd,adadelta,adam}
                        Number of steps to run model
  --optimizer_kwargs_model OPTIMIZER_KWARGS_MODEL
                        Comma separated kwargs for model optimizer. Keys and
                        values are separated by a colon. E.g.
                        lr:0.01,alpha:0.2
  --optimizer_kwargs_meta OPTIMIZER_KWARGS_META
                        Comma separated kwargs for meta-model optimizer. Keys
                        and values are separated by a colon. E.g.
                        lr:0.01,alpha:0.2
  --num_steps_model NUM_STEPS_MODEL
                        Number of steps to run model.
  --num_steps_meta NUM_STEPS_META
                        Number of steps to run meta-model.
  --reset               If set, delete the existing model directory and start
                        training from scratch.
  --supply_truth        Supply optimizer deltas to meta-optimizer.
  --run_name RUN_NAME   Name of run.
  --test TEST           Dir path for trained model/config file. THIS OVERRIDES
                        OTHER OPTIONS.
  --freq_save FREQ_SAVE
                        Number of meta steps before saving learner.
  --freq_debug FREQ_DEBUG
                        Number of meta steps before outputtig diagnostic info.
```