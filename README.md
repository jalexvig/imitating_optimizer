# Imitating Optimizers

See [this post](http://jalexvig.github.io/blog/imitating-an-optimizer/) for more details.

### Run

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --optimizer_model sgd --optimizer_kwargs_model lr:0.01 --num_steps_meta 10000 --seed 1 --num_steps_model 5 --freq_debug 100 --supply_truth
```