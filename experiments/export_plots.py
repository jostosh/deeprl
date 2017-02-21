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
import tensorflow as tf
import pprint



import plotly.plotly as py
import plotly.graph_objs as go



mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 14})
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import colorlover as cl


colorscale = cl.scales['8']['qual']['Dark2']
colorscalen = []

for c in cl.to_numeric(colorscale):
    colorscalen.append((c[0]/255., c[1]/255, c[2]/255))
colorscalen.append((0., 0., 0.))
colorscalen.append((1., 0., 0.))


def event_arrays_to_np_arrays(event_array):
    value_by_step = {}
    np_arrays_x = []
    np_arrays_y = []
    for event in event_array:
        for scalar_event in event:
            if scalar_event.step not in value_by_step:
                value_by_step[scalar_event.step] = [scalar_event.value]
            else:
                value_by_step[scalar_event.step].append(scalar_event.value)
        np_arrays_x.append(np.asarray([se.step for se in event]))
        np_arrays_y.append(np.asarray([se.value for se in event]))

    continuous_arrays_x = []
    continuous_arrays_y = []

    min_x = -np.inf
    max_x = np.inf
    for x_arr, y_arr in zip(np_arrays_x, np_arrays_y):
        xc_arr = []
        yc_arr = []
        min_x = max(min_x, x_arr[0])
        max_x = min(max_x, x_arr[-1])

        for i in range(len(x_arr) - 1):
            xc_arr.append(np.linspace(x_arr[i], x_arr[i+1], x_arr[i+1] - x_arr[i] + 1, dtype=np.int64)[:-1])
            print(xc_arr[-1])
            yc_arr.append(np.linspace(y_arr[i], y_arr[i+1], x_arr[i+1] - x_arr[i] + 1)[:-1])
        xc_arr.append(np.asarray([x_arr[-1]]))
        yc_arr.append(np.asarray([y_arr[-1]]))

        continuous_arrays_x.append(np.concatenate(xc_arr))
        continuous_arrays_y.append(np.concatenate(yc_arr))

        print(continuous_arrays_x[-1][-1] - continuous_arrays_x[-1][0], len(continuous_arrays_x[-1]),
              len(continuous_arrays_y[-1]))

    c_arrays_y = []
    for x_arr, y_arr in zip(continuous_arrays_x, continuous_arrays_y):
        idx1 = x_arr.tolist().index(min_x)
        idx2 = x_arr.tolist().index(max_x)

        #y_arr = np.asarray(y_arr[idx1:idx2])

        #print(x_arr[idx1:idx2], idx1, idx2, len(y_arr[idx1:idx2]))

        c_arrays_y.append(y_arr[idx1:idx2])
    #print(min_x, max_x)

    if args.interpolate:
        values = np.mean(np.stack(c_arrays_y), axis=0)
        errors = np.std(np.stack(c_arrays_y), axis=0)
        steps = np.arange(min_x, max_x)
        return steps, values, errors, np_arrays_x, np_arrays_y

    error_by_step = {}
    for step, val in value_by_step.items():
        error_by_step[step] = np.std(val)
        value_by_step[step] = np.mean(val)

    steps = np.asarray([k for k in value_by_step.keys()])
    values = np.asarray([v for v in value_by_step.values()])
    errors = np.asarray([v for v in error_by_step.values()])
    return steps, values, errors, np_arrays_x, np_arrays_y


def obtain_name(hp):
    function_by_name = {
        'idx': lambda p: 'Fold ' + hp['idx'],
        'model': lambda p: {'default': 'Default CNN', 'spatial': 'SIWS CNN'}[hp[p]]
    }

    if args.trace_by:
        return ' '.join(function_by_name[n](n) for n in args.trace_by)

    return hp['model'].upper().replace('_', '-') + \
        (' {}FP'.format('r' if hp['residual_prediction'] else '') if hp['frame_prediction'] else '') + \
        (' OT' if hp['optimality_tightening'] else '') + \
        (' relu' if 'activation' not in hp else hp['activation']) + \
        ' ({})'.format(hp['t_max'] if 't_max' in hp else '')


def export_plots():
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=go.XAxis(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False,
            title=args.xlabel
        ),
        yaxis=go.YAxis(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False,
            title=args.ylabel
        ),
    )
    if args.xrange:
        layout.xaxis.range = args.xrange
    if args.yrange:
        layout.xaxis.range = args.yrange

    event_files_by_hp_by_env = {}
    file_count = 0
    for root, dir, files in os.walk(args.input_dir):
        print(root, dir, files)

        if any([IsTensorFlowEventsFile(f) for f in files]) and 'hyper_parameters.pkl' in files:
            with open(os.path.join(root, 'hyper_parameters.pkl'), 'rb') as f:
                hyper_parameters = pickle.load(f)
            hyper_parameters = hyper_parameters.__dict__ if isinstance(hyper_parameters, HyperParameters) \
                else hyper_parameters
            hyper_parameters['idx'] = str(file_count)

            for param in args.ignore_params:
                if param in hyper_parameters:
                    del hyper_parameters[param]
            file_count += 1
            event_files = [os.path.join(root, f) for f in files if IsTensorFlowEventsFile(f)]

            if 'activation' in hyper_parameters:
                print(hyper_parameters['activation'].__dict__)
                hyper_parameters['activation'] = 'elu' if hyper_parameters['activation'] == tf.nn.elu else 'relu'

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

        data_objs = []
        for hyper_parameters_str, event_files in sorted(event_files_by_hp.items()):
            hyper_parameters = json.loads(hyper_parameters_str)
            events_by_scalar = {}
            print("Currently looking at {} event files".format(len(event_files)))
            pprint.pprint(hyper_parameters)
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                for scalar in ea.Tags()['scalars']:
                    if args.scalar_subset and scalar not in args.scalar_subset:
                        continue
                    #if scalar != 'Evaluation/Score':
                    #    continue
                    items = ea.Scalars(scalar)
                    print(items)
                    if scalar not in events_by_scalar:
                        events_by_scalar[scalar] = [items]
                    else:
                        events_by_scalar[scalar].append(items)

            for scalar, event_arrays in events_by_scalar.items():
                steps, values, errors, np_arrays_x, np_arrays_y = event_arrays_to_np_arrays(event_arrays)

                steps, values, errors = zip(*sorted(zip(steps, values, errors)))
                steps = np.asarray(steps)
                values = np.asarray(values)
                errors = np.asarray(errors)

                print(errors)

                plt.fill_between(steps, values - errors, values + errors, facecolor=colorscalen[hp_idx], alpha=0.2)

                handles.append(plt.plot(steps, values, linewidth=2.0, color=colorscalen[hp_idx],
                                            label=obtain_name(hyper_parameters))[0])
                trace = go.Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([values + errors, (values - errors)[::-1]]),
                    fill='tozerox',
                    fillcolor=colorscale[hp_idx].replace('rgb', 'rgba').replace(')', ',0.2)'),  # 'rgba(0,100,80,0.2)',
                    line=go.Line(color='transparent'),
                    showlegend=False
                )
                line = go.Scatter(
                    x=steps,
                    y=values,
                    line=go.Line(color=colorscale[hp_idx], width=4),
                    mode='lines',
                    name=obtain_name(hyper_parameters)
                )
                data_objs += [trace, line]

            hp_idx += 1


        position_by_env = {
            'Breakout-v0': 'upper left',
            'Pong-v0': 'lower right',
            'BeamRider-v0': 'upper left',
            'adience': 'lower right'
        }

        data = go.Data(data_objs)
        layout.title = env.replace('-v0', '') if not args.title else args.title
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=env.replace('-v0', '') + args.image_suffix + '.html')

        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
        plt.title(env.replace('-v0', '') if not args.title else args.title)
        if args.xrange:
            plt.xlim(args.xrange)
        if args.yrange:
            plt.ylim(args.yrange)
        plt.legend(handles=handles, loc=position_by_env[env], framealpha=0.)

        plt.savefig(os.path.join(args.output_dir, env.replace('-v0', '') + args.image_suffix + '.pdf'))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir", default='/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/im')
    parser.add_argument("--scalar_subset", nargs='+', default=['Evaluation/Score'])
    parser.add_argument("--ignore_params", nargs='+', default=['git_description'])
    parser.add_argument("--interpolate", dest='interpolate', action='store_true')
    parser.add_argument("--image_suffix", default="")
    parser.add_argument("--xlabel", default="Train episode")
    parser.add_argument("--ylabel", default="Score")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xrange", nargs='+', default=[], type=int)
    parser.add_argument("--yrange", nargs='+', default=[], type=int)
    parser.add_argument("--trace_by", nargs='+', default=[])
    args = parser.parse_args()

    print(args.xrange)
    export_plots()