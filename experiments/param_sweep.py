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
    'lr': log_uniform,
    'ss_epsilon': uniform,
    'global_clip_norm': log_uniform,
    'otc': log_uniform,
    'fplc': log_uniform,
    'fp_decay': log_uniform,
    'lpq_p0': log_uniform
}

lo_by_param = {
    'lr': 1e-6,
    'ss_epsilon': 0.1,
    'global_clip_norm': 0.5,
    'otc': 2.0 ** (-8),
    'fplc': 1e-5,
    'fp_decay': 0.999,
    'lpq_p0': 0.9
}

hi_by_param = {
    'lr': 1e-2,
    'ss_epsilon': 0.5,
    'global_clip_norm': 40.0,
    'otc': 2.0,
    'fplc': 1.0,
    'fp_decay': 0.99999999,
    'lpq_p0': 0.999
}

presets = {
    '0': {
        'model': 'a3c_ff',
        'optimizer': 'adam'
    },
    '1': {
        'model': 'a3c_ff',
        'optimizer': 'adam',
        'feedback': True
    },
    '2': {
        'model': 'a3c_ff_ss'
    },
    '3': {
        'model': 'a3c_ff',
        'activation': 'elu',
        'safe_softmax': True
    },
    '4': {
        'model': 'a3c_ff',
        'activation': 'elu'
    },
    '5': {
        'model': 'a3c_ff_ss',
        'safe_softmax': True
    },
    '6': {
        'model': 'a3c_ff_ss',
        'safe_softmax': True,
        'activation': 'elu'
    },
    '7': {
        'model': 'a3c_ff_ss',
        'safe_softmax': True,
        'global_clipping': True
    },
    '8': {
        'model': 'a3c_conv_lstm',
        'global_clipping': True
    },
    # -------------------------------- 08-03
    '9': {
        'model': 'a3c_ff_ss',
        'global_clipping': True
    },
    '10': {
        'model': 'a3c_ff',
        'optimizer': 'adam'
    },
    '11': {
        'model': 'a3c_ff',
        'optimizer': 'adam',
        'feedback': True
    },
    '13': {
        'model': 'a3c_ff'
    },
    '14': {
        'model': 'a3c_ff',
        'mbc': True
    },
    '15': {
        'model': 'a3c_ff_ss'
    },
    '16': {
        'model': 'a3c_conv_lstm'
    },
    '17': {
        'model': 'a3c_lstm'
    },
    '18': {
        'model': 'a3c_ff',
        'frame_prediction': True,
    },
    '19': {
        'model': 'a3c_ff',
        'frame_prediction': True,
        'residual_prediction': True
    },
    '20': {
        'model': 'a3c_ff',
        'optimality_tightening': True
    },
    '21': {
        'model': 'a3c_sisws'
    },
    '22': {
        'model': 'a3c_sisws_s'
    },
    # ---------- 18-03
    '23': {
        'model': 'a3c_ff',
        'feedback': True
    },
    '24': {
        'model': 'a3c_conv_lstm_k'
    },
    '25': {
        'model': 'a3c_ff',
        'frame_prediction': True,
        'residual_prediction': True,
        'fplc': 3.16e-5,
        'fp_decay': 0.99919,
    },
    '26': {
        'model': 'a3c_ff',
        'frame_prediction': True,
        'fplc': 0.05,
        'fp_decay': 0.9995
    },
    '12': {
        'model': 'a3c_ff',
        'global_clipping': True
    },
    # 23-03
    '27': {
        'model': 'a3c_ff_ss',
    },
    '28': {
        'model': 'a3c_ff_ss',
        'trainable_temp': True
    },
    '29': {
        'model': 'a3c_ff_ss',
        'softmax_only': True
    },
    '30': {
        'model': 'a3cglpq',
        'entropy_beta': 0.001
    },
    '31': {
        'model': 'a3cglpq',
        'entropy_beta': 0.001
    }
}


def convert_preset_to_params(preset):
    ret = ["--{}={}".format(p, v) for p, v in preset.items() if not isinstance(v, bool)]
    ret += ["--{}".format(p) for p, v in preset.items() if isinstance(v, bool) and v]
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    parser.add_argument("--params", nargs='+', default=['lr'])
    parser.add_argument("--a3c_args", nargs='+', default=[])
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    a3c_args = ['--' + a for a in args.a3c_args]

    for i in range(100):
        if args.preset:
            preset_str = convert_preset_to_params(presets[args.preset])
            command = ["python3", "mproj/deeprl/train.py", '--env=Catch', '--log_base=/home/jos/tensorflowlogs',
                       '--eval_interval=50000', '--T_max=1000000', '--n_threads=12',
                       '--log_prefix=sweep/preset{}'.format(args.preset)] + preset_str + \
                      ["--{}={}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        else:
            command = ["python3", "mproj/deeprl/rlmethods/train.py", '--env=Catch', '--model={}'.format(args.model),
                       '--evaluation_interval=50000', '--T_max=1000000',  '--n_threads=12'] + a3c_args + \
                      ["--{}={}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        print('INITIATING COMMAND:\n{}'.format(' '.join(command)))

        subprocess.run(command)
