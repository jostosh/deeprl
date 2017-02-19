import subprocess
import numpy as np
import argparse


def log_uniform(lo, hi, rate):
  log_lo = np.log(lo)
  log_hi = np.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return np.exp(v)


lo_by_param = {
    'learning_rate': 1e-4,
    'ss_epsilon': 0.1
}

hi_by_param = {
    'learning_rate': 1e-2,
    'ss_epsilon': 0.99
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    parser.add_argument("--params", nargs='+', default=['learning_rate'])
    args = parser.parse_args()

    for i in range(50):
        command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch', '--model={}'.format(args.model),
                   '--evaluation_interval=50000', '--T_max=1000000',  '--n_threads=12'] +\
                  ["--{}={}".format(p, log_uniform(lo_by_param[p], hi_by_param[p], np.random.rand()))
                   for p in args.params]
        print('INITIATING COMMAND:\n{}'.format(' '.join(command)))

        subprocess.run(command)
