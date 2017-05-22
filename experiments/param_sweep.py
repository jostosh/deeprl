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
    'ss_temp': log_uniform,
    'prototype_factor': log_uniform,
    'ppa': lambda l, h, r: int(log_uniform(l, h, r)),
    'nwp': lambda l, h, r: int(log_uniform(l, h, r)),
    'wpr': uniform
}

lo_by_param = {
    'learning_rate': 1e-6,
    'ss_epsilon': 0.1,
    'global_clip_norm': 0.5,
    'otc': 2.0 ** (-8),
    'fplc': 1e-5,
    'fp_decay': 0.999,
    'ss_temp': 0.01,
    'prototype_factor': 1,
    'ppa': 10,
    'nwp': 3,
    'wpr': 0.01
}

hi_by_param = {
    'learning_rate': 1e-2,
    'ss_epsilon': 0.5,
    'global_clip_norm': 40.0,
    'otc': 2.0,
    'fplc': 1.0,
    'fp_decay': 0.99999999,
    'ss_temp': 1.0,
    'prototype_factor': 100,
    'ppa': 200,
    'nwp': 25,
    'wpr': 1.0
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
    },
    '36': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    '37': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    # 08-04
    '38': {
        'model': 'a3c_conv_gru',
        'ignore_annealing': True
    },
    '39': {
        'model': 'a3c_ff_ss',
        'ss_hierarchical': True
    },
    '40': {
        'model': 'a3c_ff_ss',
    },
    '41': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    # 11-04
    '42': {
        'model': 'a3c_ff',
        'weights_init': 'default'
    },
    # 16-04
    '43': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    # 19-04
    '44': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    # 21-04
    '45': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'ppa': 25,
        'nwp': 10
    },
    '46': {
        'model': 'a3c_ff',
        'policy_quantization': True
    },
    # 23-04
    '47': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'ppa': 15,
        'wpr': 0.6
    },
    # 24-04
    '48': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'cor',
        'ppa': 15,
        'wpr': 0.6
    },
    '49': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'man',
        'ppa': 15,
        'wpr': 0.6
    },
    '50': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc',
        'ppa': 15,
        'wpr': 0.6
    },
    # 26-04
    '51': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'cor'
    },
    '52': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'man'
    },
    '53': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc'
    },
    '54': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc_sq'
    }, # 05-05-17
    '55': {
        'model': 'a3c_ff'
    },
    '56': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'ppa': 15,
        'wpr': 0.6,
        'pq_sim_fn': 'euc_sq'
    }, # 08-05-17
    '57': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'ppa': 15,
        'wpr': 0.6,
        'pq_sim_fn': 'euc_sq'
    },
    '58': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'ppa': 15,
        'pq_sim_fn': 'euc_sq'
    },
    '59': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'inv_euc_sq'
    },
    '60': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'inv_euc'
    }, # 11-05-2017
    '61': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc_sq',
        'pq_soft_labels': True
    },
    '62': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc_sq',
        'zpi': True
    },
    '63': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'value_quantization': True
    }, # 15-05-2017
    '64': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_cpa': True
    },
    '65': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_sim_fn': 'euc_sq',
        'pq_soft_labels': True
    },
    '66': {
        'model': 'a3c_ff',
        'weights_init': 'tflearn'
    }, # 17-05-17
    '67': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_cpa': True,
        'pq_sim_fn': 'euc_sq'
    },
    '68': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_cpa': True,
        'ppa': 15,
        'wpr': 0.25,
        'pq_sim_fn': 'euc_sq'
    },
    '69': {
        'model': 'a3c_lstm'
    },
    '70': {
        'model': 'a3c_ff',
        'value_loss_fac': 0.5
    },
    '71': {
        'model': 'a3c_lstm',
        'value_loss_fac': 0.5
    },
    '72': {
        'model': 'a3c_ff',
        'policy_quantization': True,
        'pq_cpa': True,
        'ppa': 15,
        'nwp': 1
    }
}


def convert_preset_to_params(preset):
    ret = ["--{0}={1}".format(p, v) for p, v in preset.items() if not isinstance(v, bool)]
    ret += ["--{0}".format(p) for p, v in preset.items() if isinstance(v, bool) and v]
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c_ff_ss')
    parser.add_argument("--params", nargs='+', default=['learning_rate', 'otc', 'fplc', 'fp_decay'])
    parser.add_argument("--a3c_args", nargs='+', default=[])
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    a3c_args = ['--' + a for a in args.a3c_args]

    for i in range(100):
        if args.preset:
            preset_str = convert_preset_to_params(presets[args.preset])
            command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch',
                       '--evaluation_interval=50000', '--T_max=1000000', '--n_threads=12',
                       '--logprefix=sweep/preset{0}'.format(args.preset)] + preset_str + \
                      ["--{0}={1}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        else:
            command = ["python3", "mproj/deeprl/rlmethods/a3c.py", '--env=Catch', '--model={0}'.format(args.model),
                       '--evaluation_interval=50000', '--T_max=1000000',  '--n_threads=12'] + a3c_args + \
                      ["--{0}={1}".format(p, func_by_param[p](lo_by_param[p], hi_by_param[p], np.random.rand()))
                       for p in args.params]
        print('INITIATING COMMAND:\n{0}'.format(' '.join(command)))

        try:
            subprocess.run(command)
        except:
            subprocess.call(command)