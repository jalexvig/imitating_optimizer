### Hidden state explodes

LSTM hidden state can explode (or potentially vanish?) since we necessarily only train for a certain number of timesteps, but test on an independent/arbitrary number. This can cause parameters of system to drift despite RNN output being close to theoretical parameter update (via optimizer).

In other words state doesn't have the opportunity to hurt performance very much during training, but it does during evaluation.
