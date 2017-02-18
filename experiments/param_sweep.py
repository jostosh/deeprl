import subprocess
import numpy as np
import argparse


def log_uniform(lo, hi, rate):
  log_lo = np.log(lo)
  log_hi = np.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return np.exp(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    args = parser.parse_args()

    for i in range(50):
        lr = log_uniform(1e-8, 1e-1, np.random.rand())
        print("Learning rate: {}".format(lr))

        subprocess.run(["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch', '--model={}'.format(args.model),
                        '--learning_rate={}'.format(lr), '--evaluation_interval=50000', '--T_max=1000',
                        '--n_threads=12'])
