from tensorflow.python.summary import event_accumulator
import argparse
import os
import pickle
import json
from tensorflow.python.summary.event_accumulator import IsTensorFlowEventsFile, EventAccumulator
from deeprl.common.hyper_parameters import HyperParameters
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl

mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 12})
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import colorlover as cl


colorscale = cl.scales['8']['qual']['Dark2']
colorscalen = []

for c in cl.to_numeric(colorscale):
    colorscalen.append((c[0]/255., c[1]/255, c[2]/255))
colorscalen.append((0., 0., 0.))
colorscalen.append((1., 0., 0.))

def event_arrays_to_np_arrays(event_array):
    value_by_step = {}
    for event in event_array:
        for scalar_event in event:
            if scalar_event.step not in value_by_step:
                value_by_step[scalar_event.step] = [scalar_event.value]
            else:
                value_by_step[scalar_event.step].append(scalar_event.value)

    error_by_step = {}
    for step, val in value_by_step.items():
        error_by_step[step] = np.std(val)
        value_by_step[step] = np.mean(val)

    steps = np.asarray([k for k in value_by_step.keys()])
    values = np.asarray([v for v in value_by_step.values()])
    errors = np.asarray([v for v in error_by_step.values()])
    return steps, values, errors


def obtain_name(hp):
    return hp['model'].upper().replace('_', '-') + \
        (' {}FP'.format('r' if hp['residual_prediction'] else '') if hp['frame_prediction'] else '') + \
        (' OT' if hp['optimality_tightening'] else '') + \
        ' ({})'.format(hp['t_max'])


def export_plots():

    event_files_by_hp_by_env = {}
    for root, dir, files in os.walk(args.input_dir):
        if any([IsTensorFlowEventsFile(f) for f in files]) and 'hyper_parameters.pkl' in files:
            with open(os.path.join(root, 'hyper_parameters.pkl'), 'rb') as f:
                hyper_parameters = pickle.load(f)
            hyper_parameters = hyper_parameters.__dict__ if isinstance(hyper_parameters, HyperParameters) \
                else hyper_parameters
            if 'git_description' in hyper_parameters:
                del hyper_parameters['git_description']
            event_files = [os.path.join(root, f) for f in files if IsTensorFlowEventsFile(f)]
            hyper_parameters_str = json.dumps(hyper_parameters, sort_keys=True)

            if hyper_parameters['env'] not in event_files_by_hp_by_env:
                event_files_by_hp_by_env[hyper_parameters['env']] = {hyper_parameters_str: event_files}
            elif hyper_parameters_str not in event_files_by_hp_by_env[hyper_parameters['env']]:
                event_files_by_hp_by_env[hyper_parameters['env']][hyper_parameters_str] = event_files
            else:
                event_files_by_hp_by_env[hyper_parameters['env']][hyper_parameters_str] += event_files

    for env, event_files_by_hp in event_files_by_hp_by_env.items():
        hp_idx = 0

        handles = []
        for hyper_parameters_str, event_files in sorted(event_files_by_hp.items()):
            hyper_parameters = json.loads(hyper_parameters_str)
            events_by_scalar = {}
            print("Currently looking at {} event files".format(len(event_files)))
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                for scalar in ea.Tags()['scalars']:
                    if args.scalar_subset and scalar not in args.scalar_subset:
                        continue
                    if scalar != 'Evaluation/Score':
                        continue
                    items = ea.Scalars(scalar)
                    if scalar not in events_by_scalar:
                        events_by_scalar[scalar] = [items]
                    else:
                        events_by_scalar[scalar].append(items)

            for scalar, event_arrays in events_by_scalar.items():
                steps, values, errors = event_arrays_to_np_arrays(event_arrays)

                steps, values, errors = zip(*sorted(zip(steps, values, errors)))
                steps = np.asarray(steps)
                values = np.asarray(values)
                errors = np.asarray(errors)

                hp_idx += 1

                plt.fill_between(steps, values - errors, values + errors, facecolor=colorscalen[hp_idx], alpha=0.2)

                handles.append(plt.plot(steps, values, linewidth=2.0, color=colorscalen[hp_idx],
                                            label=obtain_name(hyper_parameters))[0])
        position_by_env = {
            'Breakout-v0': 'upper left',
            'Pong-v0': 'lower right',
            'BeamRider-v0': 'upper left'
        }

        plt.xlabel('Train episode')
        plt.ylabel('Score')
        plt.title(env.replace('-v0', ''))
        plt.legend(handles=handles, loc=position_by_env[env], framealpha=0.)

        plt.savefig(os.path.join(args.output_dir, env.replace('-v0', '') + args.image_suffix + '.pdf'))
        plt.clf()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir", default='/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im')
    parser.add_argument("--scalar_subset", nargs='+', default=None)
    parser.add_argument("--image_suffix", default="")
    args = parser.parse_args()

    export_plots()