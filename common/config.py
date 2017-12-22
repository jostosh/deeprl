import argparse
import os


class Config:
    lr = 7e-4
    stacked_frames = 4
    env = dict(default='Pong', type=str, choices=['Pong', 'Breakout', 'BeamRider', 'Qbert', 'SpaceInvaders', 'Catch'])
    model = dict(default='a3cff', type=str, choices=['a3cff', 'a3clws', 'a3css', 'a3cww', 'a3clpq', 'a3cglpq'])
    T_max = int(1e8)
    t_max = 20
    gamma = 0.99
    rms_decay = 0.99
    rms_epsilon = 0.1
    entropy_beta = 0.01
    n_threads = 16
    render = False
    log_prefix = ''
    eval_interval = int(1e6)
    stat_interval = 100
    score_interval = 50
    eval_episodes = 25
    weight_init = 'glorot'
    async = False

    ppa = 16
    lpq_distance_fn = 'euc2'
    lpq_temp = 1.0
    lpq_p0 = 0.9
    lpq_pN = 0.99
    lpq_init = 'glorot'

    im_w = 84
    im_h = 84

    version = 'v1.0'

    log_base = os.path.join(os.path.dirname(__file__), '../logs')
    log_dir = None



def _get_log_dir():
    path = os.path.join(Config.log_base, Config.version, Config.log_prefix, Config.env, Config.model)
    os.makedirs(path, exist_ok=True)
    # Check the current directories in there
    current_dirs = sorted([o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))])
    # We're assuming at this point that we do not exceed 1M runs per version
    if not current_dirs:
        # If there are no dirs yet, start at 0
        rundir = 'run000000'
    else:
        # Otherwise make a new one by incrementing the count
        lastdir     = current_dirs[-1]
        lastrun     = int(lastdir[3:])
        rundir      = "run%06d" % (lastrun + 1,)
    fulldir = os.path.join(path, rundir)
    return fulldir


def load_config():
    parser = argparse.ArgumentParser()
    for arg, default in Config.__dict__.items():
        if isinstance(default, bool):
            parser.add_argument("--" + arg, action='store_true', dest=arg)
            parser.add_argument("--not_" + arg, action='store_false', dest=arg)
            parser.set_defaults(**{arg: default})
        else:
            if isinstance(default, dict):
                parser.add_argument("--" + arg, **default)
            else:
                parser.add_argument("--" + arg, type=type(default), default=default)

    args = parser.parse_args()
    for key, val in vars(args).items():
        setattr(Config, key, val)

    Config.log_dir = _get_log_dir()
    return Config
