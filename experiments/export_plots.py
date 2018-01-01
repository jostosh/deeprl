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

from matplotlib.mlab import griddata
import plotly.plotly as py
import plotly.graph_objs as go
from copy import deepcopy
from matplotlib import cm

import colorlover as cl
from scipy.interpolate import spline
import re
from scipy.signal import savgol_filter


colorscale = cl.scales['8']['qual']['Dark2']
colorscalen = []

for c in cl.to_numeric(colorscale):
    colorscalen.append((c[0]/255., c[1]/255, c[2]/255))
colorscalen.append((0., 0., 0.))
colorscalen.append((1., 0., 0.))


def event_arrays_to_mean_and_errors(event_array):
    value_by_step = {}
    np_arrays_x = []
    np_arrays_y = []
    for event in event_array:
        for i, scalar_event in enumerate(event):
            step = i+1 if args.step_to_epoch else scalar_event.step
            if step not in value_by_step:
                value_by_step[step] = [scalar_event.value]
            else:
                value_by_step[step].append(scalar_event.value)
        np_arrays_x.append(np.asarray([i+1 if args.step_to_epoch else se.step for i, se in enumerate(event)]))
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
            yc_arr.append(np.linspace(y_arr[i], y_arr[i+1], x_arr[i+1] - x_arr[i] + 1)[:-1])
        xc_arr.append(np.asarray([x_arr[-1]]))
        yc_arr.append(np.asarray([y_arr[-1]]))

        continuous_arrays_x.append(np.concatenate(xc_arr))
        continuous_arrays_y.append(np.concatenate(yc_arr))

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
        errors = np.std(np.stack(c_arrays_y), axis=0) / np.sqrt(len(c_arrays_y))
        steps = np.arange(min_x, max_x)
        return steps, values, errors, np_arrays_x, np_arrays_y

    error_by_step = {}
    for step, val in value_by_step.items():
        error_by_step[step] = np.std(val) / np.sqrt(len(val))
        value_by_step[step] = np.mean(val)

    steps = np.asarray([k for k in value_by_step.keys()])
    values = np.asarray([v for v in value_by_step.values()])
    errors = np.asarray([v for v in error_by_step.values()])
    return steps, values, errors, np_arrays_x, np_arrays_y


def obtain_name(hp):
    function_by_name = {
        'idx': lambda p: 'Fold ' + hp['idx'],
        'model': lambda p: {
            'default': 'Default CNN',
            'spatial': 'SISWS default',
            'per_feature': "SISWS PF",
            'centroids_trainable': 'LWS small',
            'large': 'LWS',
            'inv_euclidean': 'SISWS IE',
            'a3c_ff': 'A3C Softmax',
            'a3c_ff_ss': 'A3C SS',
            'a3c_sisws': 'A3C LWS',
            'a3c_ff_ww': 'A3C WW',
            'a3c_lstm': "A3C LSTM"
        }[hp[p]],
        'per_feature': lambda p: '/F' if (p in hp and hp[p] == True) else '',
        'policy_quantization': lambda p: "PQ" if (hp[p] == True) else "",
        'value_loss_fac': lambda p: ' $\lambda_V = {}$'.format(hp[p]),
        'zpi': lambda p: ''
    }

    print(hp)
    if 'policy_quantization' in hp and hp['policy_quantization'] and 'glvq' in hp and hp['glvq'] and 'zpi' in hp and \
            hp['zpi']:
        return 'A3C GLPQ zpi'
    if 'policy_quantization' in hp and hp['policy_quantization'] and 'glvq' in hp and hp['glvq'] and 'lpq_hot' in hp and hp['lpq_hot']:
        return 'A3C GLPQ hot'
    if 'policy_quantization' in hp and hp['policy_quantization'] and 'glvq' in hp and hp['glvq']:
        return 'A3C GLPQ'
    elif 'policy_quantization' in hp and hp['policy_quantization']:
        return 'A3C LPQ'


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
        layout.yaxis.range = args.yrange

    if len(args.input_dir) > 1:
        handles = []
        data_by_label = {}

        for label, input_dir in zip(args.labels, args.input_dir):
            event_files_by_hp_by_env = get_event_files_by_hp_by_env(input_dir)

            for env, event_files_by_hp in event_files_by_hp_by_env.items():
                hp_idx = 0

                all_scores = []
                all_xticks = []
                all_yticks = []
                all_surfaces = []
                for hyper_parameters_str, event_files in sorted(event_files_by_hp.items()):
                    hyper_parameters = json.loads(hyper_parameters_str)
                    #print("Currently looking at {} event files".format(len(event_files)))
                    pprint.pprint(hyper_parameters)

                    events_by_scalar = get_events_by_scalar(event_files)


                    for scalar, event_arrays in events_by_scalar.items():
                        steps, values, errors, np_arrays_x, np_arrays_y = event_arrays_to_mean_and_errors(event_arrays)

                        #print(hyper_parameters_str, len(steps))

                        all_scores, all_surfaces, all_xticks, all_yticks = get_sweep_data(all_scores, all_surfaces,
                                                                                          all_xticks, all_yticks,
                                                                                          hyper_parameters, np_arrays_y)

                    hp_idx += 1

                data_by_label[label] = (all_scores, all_surfaces, all_xticks, all_yticks)

        fig, ax = plt.subplots()
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.set_title(args.title)
        ax.set_xlim(args.xrange)
        ax.set_ylim(args.yrange)

        totals = []
        for label, (scores, surfaces, xticks, yticks) in sorted(data_by_label.items()):
            xi = np.linspace(min(xticks), max(xticks))

            indices = np.argsort(xticks).astype('int64')
            yi = spline([xticks[i] for i in indices], [surfaces[i] for i in indices], xi, order=1)
            totals.append((label, np.mean(yi[12:])))

            ysmoothed = savgol_filter(yi, window_length=5, polyorder=3)

            handles.append(plt.plot(xi, np.nan_to_num(ysmoothed), linewidth=4.0, label=label)[0])
            #handles.append(plt.scatter(xticks, scores, label=args.labels[label_idx] + " Data"))

        plt.legend(handles=handles, loc=args.legend_at, framealpha=0.)

        plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '.pdf'))
        plt.gcf().patch.set_alpha(0.0)
        plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '.png'), rasterized=True, dpi=300)
        if args.display:
            plt.show()

        #plot_hbar(totals)


        plt.clf()
        handles.clear()
        fig, ax = plt.subplots()
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel('Mean score final 5 epochs')
        ax.set_title(args.title)
        ax.set_xlim(args.xrange)
        ax.set_ylim(args.yrange)

        totals = []
        for label, (scores, surfaces, xticks, yticks) in sorted(data_by_label.items()):
            xi = np.linspace(min(xticks), max(xticks))

            indices = np.argsort(xticks)
            yi = spline([xticks[i] for i in indices], [scores[i] for i in indices], xi, order=1)
            totals.append((label, np.mean(yi[12:]), np.std(yi[12:])))

            ysmoothed = savgol_filter(yi, window_length=5, polyorder=3)

            handles.append(plt.plot(xi, ysmoothed, linewidth=3.0, label=label)[0])
            #handles.append(plt.scatter(xticks, scores, label=args.labels[label_idx] + " Data"))


        plt.legend(handles=handles, loc=args.legend_at, framealpha=0.)

        plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '_scores.pdf'))
        plt.gcf().patch.set_alpha(0.0)
        plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '_scores.png'), rasterized=True, dpi=600)
        if args.display:
            plt.show()

        plot_hbar(totals)

        return

    input_dir = args.input_dir[0]
    label_idx = 0

    event_files_by_hp_by_env = get_event_files_by_hp_by_env(input_dir)

    for env, event_files_by_hp in event_files_by_hp_by_env.items():
        hp_idx = 0

        handles = []

        data_objs = []

        all_scores = []
        all_xticks = []
        all_yticks = []
        all_surfaces = []
        for hyper_parameters_str, event_files in sorted(event_files_by_hp.items()):
            hyper_parameters = json.loads(hyper_parameters_str)
            print("Currently looking at {} event files".format(len(event_files)))
            pprint.pprint(hyper_parameters)

            events_by_scalar = get_events_by_scalar(sorted(event_files))

            for scalar, event_arrays in events_by_scalar.items():
                steps, values, errors, np_arrays_x, np_arrays_y = event_arrays_to_mean_and_errors(event_arrays)

                if args.export_best:
                    best_index = np.argmax(values)

                    with open(os.path.join(args.output_dir, 'best_{}.txt'
                            .format(hyper_parameters[args.trace_by[0]])), 'w') as f:
                        [f.write(str(arr[best_index]) + '\n') for arr in np_arrays_y]


                if args.mode == 'mean':
                    steps, values, errors = zip(*sorted(zip(steps, values, errors)))
                    steps = np.asarray(steps)
                    values = np.asarray(values)
                    errors = np.asarray(errors)

                    add_to_mpl_plot(errors, handles, hp_idx, hyper_parameters, steps, values)
                    add_to_plotly_plot(data_objs, errors, hp_idx, hyper_parameters, steps, values)

                elif args.mode == 'sweep':
                    all_scores, all_surfaces, all_xticks, all_yticks = get_sweep_data(all_scores, all_surfaces,
                                                                                      all_xticks, all_yticks,
                                                                                      hyper_parameters, np_arrays_y)

            hp_idx += 1


        position_by_env = {
            'Breakout-v0': 'upper left',
            'Pong-v0': 'lower right',
            'BeamRider-v0': 'upper left',
            'adience': 'lower right'
        }

        if args.mode == 'mean':
            #render_mean_score_plotly(data_objs, env, layout)
            render_mean_score_mpl(env, handles, position_by_env)
        else:
            xlab = args.trace_by[0].split('_').title() if not args.xlabel else args.xlabel
            ylab = args.scalar_subset[0].split('/')[-1].title() if not args.ylabel else args.ylabel
            if len(args.trace_by) == 1:
                title = env.replace('-v0', '') if not args.title else args.title

                render_sweep1d_mpl(all_scores, all_surfaces, all_xticks, title, xlab, ylab)
                render_sweep1d_plotly(all_scores, all_surfaces, all_xticks, env, layout, title, xlab, ylab)
            else:
                render_sweep2d_mpl(all_scores, all_xticks, all_yticks, env, xlab, ylab)


def plot_hbar(totals):
    newsize = (args.fontsize * 2) // 3
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': newsize})
    mpl.rc('xtick', labelsize=newsize)
    mpl.rc('ytick', labelsize=newsize)

    fig, ax = plt.subplots(figsize=(6, len(totals) + 1))
    width = 0.75
    ind = np.arange(len(totals))

    ax.barh(ind, [t[1] for t in totals], width, color='blue')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels([t[0] for t in totals], minor=False)
    for i, (_, v, s) in enumerate(totals):
        ax.text(v + 0.01, i + 0.3, "$%.2f \pm %.2f$" % (v, s), color='blue')

    ax.set_xlim([0.2, 0.9])
    plt.title(args.title)
    plt.xlabel('Mean score final 5 epochs over all runs')
    plt.ylabel('Method')

    plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '_bars.pdf'))
    plt.gcf().patch.set_alpha(0.0)
    plt.savefig(os.path.join(args.output_dir, args.title.lower().replace(' ', '_') + '_bars.png'), rasterized=True, dpi=600)

    if args.display:
        plt.show()


def render_sweep2d_mpl(all_scores, all_xticks, all_yticks, env, xlab, ylab):
    from mpl_toolkits.mplot3d import axes3d
    min_x, max_x = np.min(all_xticks), np.max(all_xticks)
    min_y, max_y = np.min(all_yticks), np.max(all_yticks)
    xi = np.linspace(min_x, max_x)
    yi = np.linspace(min_y, max_y)
    zi = griddata(all_xticks, all_yticks, all_scores, xi, yi, interp='nn')
    title = env.replace('-v0', '') if not args.title else args.title
    fig, ax = plt.subplots()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if args.xrange:
        ax.set_xlim(args.xrange)
    if args.yrange:
        ax.set_ylim(args.yrange)
    CS = ax.contour(xi, yi, zi, args.levels, linewidths=0.5, colors='k')
    CS = ax.contourf(xi, yi, zi, args.levels, cmap=cm.viridis)
    sc = ax.scatter(all_xticks, all_yticks, c='k')
    cb = fig.colorbar(CS)
    cb.ax.set_title("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, title.lower().replace(' ', '_') + '.pdf'))
    plt.gcf().patch.set_alpha(0.0)
    plt.savefig(os.path.join(args.output_dir, title.lower().replace(' ', '_') + '.png'), rasterized=True, dpi=600)
    if args.display:
        plt.show()
    plt.clf()


def render_sweep1d_mpl(all_scores, all_surfaces, all_xticks, title, xlab, ylab):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if args.xrange:
        ax.set_xlim(args.xrange)
    if args.yrange:
        ax.set_ylim(args.yrange)
    cax = ax.scatter(all_xticks, all_scores, s=80, c=all_surfaces, cmap=cm.winter)
    cb = fig.colorbar(cax)
    cb.ax.set_title("$\sum_i s_i$")
    #plt.tight_layout()
    #ax.tight_layout()
    plt.savefig(os.path.join(args.output_dir, title.lower().replace(' ', '_').replace('.', '').replace(',', '') + '.pdf'))
    plt.gcf().patch.set_alpha(0.0)
    plt.savefig(os.path.join(args.output_dir, title.lower().replace(' ', '_').replace('.', '').replace(',', '') + '.png'))
    plt.show()
    plt.clf()


def render_sweep1d_plotly(all_scores, all_surfaces, all_xticks, env, layout, title, xlab, ylab):
    trace = go.Scatter(
        x=all_xticks,
        y=all_scores,
        mode='markers',
        marker=go.Marker(
            size='14',
            color=all_surfaces,
            colorscale='Viridis',
            showscale=True
        )
    )
    layout.xaxis.title = xlab
    # layout.xaxis.type = 'log'
    layout.yaxis.title = ylab
    layout.title = title
    data = go.Data([trace])
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=env.replace('-v0', '' if not args.title else args.title))


def render_mean_score_mpl(env, handles, position_by_env):
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(env.replace('-v0', '') if not args.title else args.title)
    if args.xrange:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)
    plt.legend(handles=handles, loc=args.legend_at, framealpha=0.)
    plt.savefig(os.path.join(args.output_dir, env.replace('-v0', '') + args.image_suffix + '.pdf'))
    plt.gcf().patch.set_alpha(0.0)
    plt.savefig(os.path.join(args.output_dir, env.replace('-v0', '') + args.image_suffix + '.png'), dpi=600)

    plt.clf()


def render_mean_score_plotly(data_objs, env, layout):
    data = go.Data(data_objs)
    layout.title = env.replace('-v0', '') if not args.title else args.title
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=env.replace('-v0', '') + args.image_suffix + '.html')

all_tick_combinations = []

def get_sweep_data(all_scores, all_surfaces, all_xticks, all_yticks, hyper_parameters, np_arrays_y):
    if len(args.trace_by) == 1:
        all_scores += [np.mean(y_arr[-5:]) for y_arr in np_arrays_y]
        all_surfaces += [np.mean(y_arr) for y_arr in np_arrays_y]

        transform = np.log10 if not args.no_log_scale else lambda x: x

        all_xticks += [transform(hyper_parameters[args.trace_by[0]]) for _ in range(len(np_arrays_y))]
    elif len(args.trace_by) == 2:

        if ((hyper_parameters[args.trace_by[0]], hyper_parameters[args.trace_by[1]]) not in all_tick_combinations):
            all_scores += [np.mean(y_arr[-5:]) for y_arr in np_arrays_y]
            all_surfaces += [np.mean(y_arr) for y_arr in np_arrays_y]

            all_xticks += [np.log10(hyper_parameters[args.trace_by[0]]) for _ in range(len(np_arrays_y))]

            fun = np.log10 if args.log_scale else lambda s: s
            all_yticks += [fun(hyper_parameters[args.trace_by[1]]) for _ in range(len(np_arrays_y))]

            all_tick_combinations.append((hyper_parameters[args.trace_by[0]], hyper_parameters[args.trace_by[1]]))

    else:
        raise ValueError("No support for sweep plot for more than two parameters")
    return all_scores, all_surfaces, all_xticks, all_yticks


def add_to_plotly_plot(data_objs, errors, hp_idx, hyper_parameters, steps, values):
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


def add_to_mpl_plot(errors, handles, hp_idx, hyper_parameters, steps, values):
    plt.fill_between(steps, values - errors, values + errors, facecolor=colorscalen[hp_idx], alpha=0.2)
    handles.append(plt.plot(steps, values, linewidth=2.0, color=colorscalen[hp_idx],
                            label=obtain_name(hyper_parameters))[0])


def get_events_by_scalar(event_files):
    events_by_scalar = {}
    for event_file in event_files:
        print(event_file)
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        for scalar in ea.Tags()['scalars']:
            if args.scalar_subset and scalar not in args.scalar_subset:
                continue
            # if scalar != 'Evaluation/Score':
            #    continue
            items = ea.Scalars(scalar)
            if scalar not in events_by_scalar:
                events_by_scalar[scalar] = [items]
            else:
                events_by_scalar[scalar].append(items)
    return events_by_scalar


def get_event_files_by_hp_by_env(input_dir):
    event_files_by_hp_by_env = {}
    file_count = 0
    for root, dir, files in os.walk(input_dir):
        #print(root, dir, files)

        if any([IsTensorFlowEventsFile(f) for f in files]) and 'hyper_parameters.pkl' in files:
            try:
                with open(os.path.join(root, 'hyper_parameters.pkl'), 'rb') as f:
                    hyper_parameters = pickle.load(f)
            except Exception as e:
                continue
            hyper_parameters = hyper_parameters.__dict__ if isinstance(hyper_parameters, HyperParameters) \
                else hyper_parameters
            hyper_parameters['idx'] = str(file_count)

            # if any(t not in hyper_parameters for t in args.trace_by):
            #    continue
            if 'value_loss_fac' not in hyper_parameters:
                hyper_parameters['value_loss_fac'] = 1.0

            for param in args.ignore_params:
                if param in hyper_parameters:
                    del hyper_parameters[param]
            file_count += 1
            event_files = [os.path.join(root, f) for f in files if IsTensorFlowEventsFile(f)]

            if 'activation' in hyper_parameters:
                #print(hyper_parameters['activation'].__dict__)
                hyper_parameters['activation'] = 'elu' if hyper_parameters['activation'] == tf.nn.elu else 'relu'

            if args.subset_params:
                hyper_parameters = {p: hyper_parameters[p] for p in args.subset_params if p in hyper_parameters}

            if hyper_parameters['model'] in args.exclude or any(a in hyper_parameters and hyper_parameters[a] == True for a in args.exclude):
                continue

            hyper_parameters_str = json.dumps(hyper_parameters, sort_keys=True)

            if hyper_parameters['env'] not in event_files_by_hp_by_env:
                event_files_by_hp_by_env[hyper_parameters['env']] = {hyper_parameters_str: event_files}
            elif hyper_parameters_str not in event_files_by_hp_by_env[hyper_parameters['env']]:
                event_files_by_hp_by_env[hyper_parameters['env']][hyper_parameters_str] = event_files
            else:
                event_files_by_hp_by_env[hyper_parameters['env']][hyper_parameters_str] += event_files
        elif 'hyper_parameters.pkl' in files and all('fold{}'.format(i) in dir for i in range(args.folds)):
            try:
                with open(os.path.join(root, 'hyper_parameters.pkl'), 'rb') as f:
                    hyper_parameters = pickle.load(f)
            except Exception as e:
                continue

            for r, _, fs in os.walk(root):
                event_files = [os.path.join(root, r, f) for f in fs if IsTensorFlowEventsFile(f) and
                               any('fold{}'.format(i) in r for i in range(args.folds))]
                hp = deepcopy(hyper_parameters)

                searched = re.search('adience\/([A-Za-z_]*)\/', r)
                if not searched:
                    continue
                hp['model'] = searched.group(1)

                if hp['model'] in args.exclude:
                    continue
                hyper_parameters_str = json.dumps(hp, sort_keys=True)

                if 'adience' not in event_files_by_hp_by_env:
                    event_files_by_hp_by_env['adience'] = {hyper_parameters_str: event_files}
                elif hyper_parameters_str not in event_files_by_hp_by_env['adience']:
                    event_files_by_hp_by_env['adience'][hyper_parameters_str] = event_files
                else:
                    event_files_by_hp_by_env['adience'][hyper_parameters_str] += event_files
    return event_files_by_hp_by_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", nargs='+', default=[])
    parser.add_argument("--output_dir", default='/home/jos/Dropbox/RUG/6e Jaar/mproj/thesis/lpq/im')
    parser.add_argument("--scalar_subset", nargs='+', default=['Evaluation/Score'])
    parser.add_argument("--ignore_params", nargs='+', default=['git_description'])
    parser.add_argument("--subset_params", nargs='+', default=[])
    parser.add_argument("--labels", nargs='+', default=[])
    parser.add_argument("--interpolate", dest='interpolate', action='store_true')
    parser.add_argument("--log_scale", dest='log_scale', action='store_true')
    parser.add_argument("--no_log_scale", dest='no_log_scale', action='store_true')
    parser.add_argument("--display", dest='display', action='store_true')
    parser.add_argument("--image_suffix", default="")
    parser.add_argument("--xlabel", default="Train episode")
    parser.add_argument("--ylabel", default="Score")
    parser.add_argument("--zlabel", default="Score")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xrange", nargs='+', default=[], type=float)
    parser.add_argument("--yrange", nargs='+', default=[], type=float)
    parser.add_argument("--trace_by", nargs='+', default=[])
    parser.add_argument("--mode", default='mean', choices=['mean', 'sweep'])
    parser.add_argument("--legend_at", default='upper right')
    parser.add_argument("--levels", default=20, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--exclude", default=[], nargs='+')
    parser.add_argument("--step_to_epoch", dest='step_to_epoch', action='store_true')
    parser.add_argument("--export_best", dest='export_best', action='store_true')
    parser.add_argument("--fontsize", type=int, default=14)
    args = parser.parse_args()

    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': args.fontsize})
    mpl.rc('xtick', labelsize=args.fontsize)
    mpl.rc('ytick', labelsize=args.fontsize)
    mpl.rc('figure', autolayout=True)

    print(args.input_dir)
    export_plots()