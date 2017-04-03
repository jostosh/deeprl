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
    'ss_epsilon': uniform,
    'global_clip_norm': log_uniform,
    'otc': log_uniform,
    'fplc': log_uniform,
    'fp_decay': log_uniform,
    'ss_temp': log_uniform
}

lo_by_param = {
    'learning_rate': 1e-6,
    'ss_epsilon': 0.1,
    'global_clip_norm': 0.5,
    'otc': 2.0 ** (-8),
    'fplc': 1e-5,
    'fp_decay': 0.999,
    'ss_temp': 0.01
}

hi_by_param = {
    'learning_rate': 1e-2,
    'ss_epsilon': 0.5,
    'global_clip_norm': 40.0,
    'otc': 2.0,
    'fplc': 1.0,
    'fp_decay': 0.99999999,
    'ss_temp': 1.0
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
    # 30-03
    '30': {
        'model': 'a3c_ff_ss',
        'trainable_temp': True,
        'force_store': True
    },
    # 31-03
    '31': {
        'model': 'a3c_ff',
        'global_clipping': True,
        'global_clip_norm': 21.3796
    },
    '32': {
        'model': 'a3c_conv_lstm_k',
        'global_clipping': True,
        'global_clip_norm': 21.3796
    },
    # 02-04
    '33': {
        'model': 'a3c_ff_ss',
        'trainable_temp': True
    },
    '34': {
        'model': 'a3c_ff_ss'
    },
    '35': {
        'model': 'a3c_conv_gru'
    }
}


def convert_preset_to_params(preset):
    ret = ["--{}={}".format(p, v) for p, v in preset.items() if not isinstance(v, bool)]
    ret += ["--{}".format(p) for p, v in preset.items() if isinstance(v, bool) and v]
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    parser.add_argument("--params", nargs='+', default=['learning_rate', 'otc', 'fplc', 'fp_decay', 'ss_temp'])
    parser.add_argument("--a3c_args", nargs='+', default=[])
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    a3c_args = ['--' + a for a in args.a3c_args]

    for i in range(100):
        if args.preset:
            preset_str = convert_preset_to_params(presets[args.preset])
            command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch',
                       '--evaluation_interval=50000', '--T_max=1000000', '--n_threads=12',
                       '--logprefix=sweep/preset{}'.format(args.preset)] + preset_str + \
                      ["--{}={}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        else:
            command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch', '--model={}'.format(args.model),
                       '--evaluation_interval=50000', '--T_max=1000000',  '--n_threads=12'] + a3c_args + \
                      ["--{}={}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        print('INITIATING COMMAND:\n{}'.format(' '.join(command)))

        subprocess.run(command)
