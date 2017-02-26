import subprocess
import numpy as np
import argparse


def log_uniform(lo, hi, rate):
  log_lo = np.log(lo)
  log_hi = np.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return np.exp(v)


def uniform(lo, hi, rate):
    return rate * (hi - lo) + lo

func_by_param = {
    'learning_rate': log_uniform,
    'ss_epsilon': uniform
}

lo_by_param = {
    'learning_rate': 1e-6,
    'ss_epsilon': 0.1
}

hi_by_param = {
    'learning_rate': 1e-2,
    'ss_epsilon': 0.99
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    parser.add_argument("--params", nargs='+', default=['learning_rate', 'ss_epsilon'])
    parser.add_argument("--a3c_args", nargs='+', default=[])
    args = parser.parse_args()

    a3c_args = ['--' + a for a in args.a3c_args]

    for i in range(100):
        command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch', '--model={}'.format(args.model),
                   '--evaluation_interval=50000', '--T_max=1000000',  '--n_threads=12'] + a3c_args + \
                  ["--{}={}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                   for p in args.params]
        print('INITIATING COMMAND:\n{}'.format(' '.join(command)))

        subprocess.run(command)
