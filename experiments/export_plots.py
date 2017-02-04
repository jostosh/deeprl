from tensorflow.python.summary import event_accumulator
import argparse
import os
import pickle
from deeprl.common.primitive_utils import sorted_dict
import json
import pprint
from tensorflow.python.summary.event_accumulator import IsTensorFlowEventsFile, EventAccumulator
from deeprl.common.hyper_parameters import HyperParameters
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import plotly.offline as py
import time
from plotly.graph_objs import *
import colorlover as cl

colorscale = cl.scales['12']['qual']['Paired']

layout = Layout(
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(229,229,229)',
    xaxis=XAxis(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
    yaxis=YAxis(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
)

def event_arrays_to_np_arrays(event_array):
    value_by_step = {}
    for event in event_array:
        for scalar_event in event:
            if scalar_event.step not in value_by_step:
                value_by_step[scalar_event.step] = scalar_event.value

    error_by_step = {}
    for step, val in value_by_step.items():
        error_by_step[step] = np.std(val)
        value_by_step[step] = np.mean(val)

    steps = np.asarray([k for k in value_by_step.keys()])
    values = np.asarray([v for v in value_by_step.values()])
    errors = np.asarray([v for v in error_by_step.values()])
    return steps, values, errors


def export_plots():

    event_files_by_hp_by_env = {}
    for root, dir, files in os.walk(args.input_dir):
        if any([IsTensorFlowEventsFile(f) for f in files]) and 'hyper_parameters.pkl' in files:
            with open(os.path.join(root, 'hyper_parameters.pkl'), 'rb') as f:
                hyper_parameters = pickle.load(f)
            hyper_parameters = hyper_parameters.__dict__ if isinstance(hyper_parameters, HyperParameters) \
                else hyper_parameters
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

        data_objs = []
        for hyper_parameters_str, event_files in event_files_by_hp.items():
            hyper_parameters = json.loads(hyper_parameters_str)
            events_by_scalar = {}
            print("Currently looking at {} event files".format(len(event_files)))
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                for scalar in ea.Tags()['scalars']:
                    if args.scalar_subset and scalar not in args.scalar_subset:
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

                print("Now considering {}".format(scalar))
                pprint.pprint(hyper_parameters)

                trace = Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([values+errors, (values-errors)[::-1]]),
                    fill='tozerox',
                    fillcolor=colorscale[hp_idx].replace('rgb', 'rgba').replace(')', ',0.2)'),#'rgba(0,100,80,0.2)',
                    line=Line(color='transparent'),
                    showlegend=False
                )
                line = Scatter(
                    x=steps,
                    y=values,
                    line=Line(color=colorscale[hp_idx], width=4),
                    mode='lines'
                )
                data_objs += [trace, line]
                hp_idx += 1

        data = Data(data_objs)
        fig = Figure(data=data, layout=layout)
        py.plot(fig)
        time.sleep(5)


                #plt.rc('font', family='serif')
                #plt.plot(steps, values, 'k-')
                #plt.fill_between(steps, values - errors, values + errors, facecolor='green')
                #plt.grid(True, color='w', linestyle='-', linewidth=2)
                #plt.gca().patch.set_facecolor('0.8')
                #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--scalar_subset", nargs='+', default=None)
    args = parser.parse_args()

    export_plots()