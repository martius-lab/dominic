import numpy as np
import os
from glob import glob
import smart_settings
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import collections
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import h5py
    

def replace_lists_with_nums(d):
    dd = {}
    if isinstance(d, list):
        for i, vv in enumerate(d):
            dd[str(i)] = replace_lists_with_nums(vv)
        return dd
    elif hasattr(d, 'keys'):
        for k, v in d.items():
            dd[k] = replace_lists_with_nums(v)
        return dd
    else:
        return d

def tabulate_events(dpath):

    experiments = defaultdict(list)
    for root, dirs, files in os.walk(dpath):
        summary_iterators = [EventAccumulator(os.path.join(root, dname)).Reload() for dname in filter(lambda x: 'events' in x, dirs)]
        if len(summary_iterators) == 0: continue
        print("Root:", root, len(summary_iterators))
        tags = summary_iterators[0].Tags()['scalars']

        for it in summary_iterators:
            assert it.Tags()['scalars'] == tags

        out = defaultdict(list)
        steps = []

        for tag in tags:
            steps = [e.step for e in summary_iterators[0].Scalars(tag)]

            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                assert len(set(e.step for e in events)) == 1

                out[tag].append([e.value for e in events])
        experiments[root] = steps, out
    return out, steps

def list_experiments(experiment_dir):
    return os.listdir(experiment_dir)

def get_metrics_and_settings_paths(root_paths):
    metrics_paths = []
    settings_paths = []
    for root_path in tqdm(root_paths):
        metrics_paths.extend(glob(root_path+"/*/*/metrics.csv"))
        settings_paths.extend(glob(root_path+"/*/*/settings.json"))
    
    assert len(metrics_paths) == len(settings_paths)
    
    print("Num experiments:", len(metrics_paths))
    return metrics_paths, settings_paths


def load_metrics_and_settings(metrics_paths, settings_paths):
    settings = [fd(smart_settings.load(s)) for s in settings_paths]
    metrics  = [ pd.DataFrame.to_dict(pd.read_csv(m), orient='r')[0] for m in metrics_paths]
    # join metrics and flattened params
    settings_df = pd.DataFrame.from_records(settings)
    metrics_df = pd.DataFrame.from_records(metrics)

    joined_metrics_and_settings_df = pd.concat([settings_df, metrics_df], axis=1)

    return settings_df, metrics_df, joined_metrics_and_settings_df


def get_parameters_of_interest(settings_df):
    # find columns with more than 1 unique element
    params_of_interest  = []
    for col in settings_df.columns:
        if len(settings_df[col].unique())>1 and col not in ['working_dir', 'id']:
            params_of_interest.append(col)


def get_data_stats(params_of_interest, metrics_df, data_df):
    df =  data_df[params_of_interest+list(metrics_df.columns) + ['working_dir']].dropna()
    grouped_df = df[params_of_interest+list(metrics_df.columns)].groupby(params_of_interest)
    mu_df = grouped_df.mean()
    median_df = grouped_df.median()
    var_df = grouped_df.var()

    return grouped_df, mu_df, median_df, var_df


def flatten(d, parent_key='', sep='.'):
    if not hasattr(d, 'items'):
        return {parent_key: d}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                items.extend(flatten(vv, new_key+sep+str(i), sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
fd = flatten


class ExperimentSummary:
    def __init__(self, experiments, exclude_params=[]) -> None:
        self.experiments = experiments
        self._exclude_params = exclude_params
        self.settings_df = pd.DataFrame.from_records([dict(**fd(e.settings_flat)) for i, e in enumerate(experiments)])
        if hasattr(experiments[0], 'metrics'):
            self.metrics_df = pd.DataFrame.from_records( [e.metrics for e in experiments])
            self.summary_df = pd.concat([self.settings_df, self.metrics_df], axis=1)
            self._infer_params_of_interest()
            self.unique_params = self.settings_df.set_index(self.params_of_interest)
        else:
            self.metrics_df = None
            self.summary_df = self.settings_df
            self._infer_params_of_interest()
            self.unique_params = self.settings_df.set_index(self.params_of_interest)

    def _infer_params_of_interest(self):
        # find columns with more than 1 unique element
        self.params_of_interest  = []
        for col in self.settings_df.columns:
            if len(self.settings_df[col].unique())>1 and col not in ['working_dir', 'id', 'seed'] + self._exclude_params:
                self.params_of_interest.append(col)

    def print_unique_params(self):
        indexed_df = self.settings_df.set_index(self.params_of_interest)
        print(indexed_df.index)

    def unique_indexes(self):
        return [name for name, group in self._group_by_params(self.params_of_interest)]


    def print_values(self):
        print(self.experiments[0].values.keys())

    def _group_by_params(self, params_of_interest):
        if self.metrics_df is None:
            return self.summary_df[params_of_interest].groupby(params_of_interest)
        else:
            group = self.summary_df[params_of_interest+list(self.metrics_df.columns)].groupby(params_of_interest)
            return group


    def _check_contains(self, name, idxs):
        for idx in idxs: 
            is_in = True
            for k,v in zip(name, idx):
                if v is not None and k != v:
                    is_in = False
                    break
            if is_in:
                return True
        return is_in


    def time_statistics(self, value, filter_params=[], use_min_len=False, verbose=True, apply_func=lambda x:x):
        res = {}
        for name, group in self._group_by_params(self.params_of_interest):
            if filter_params is None or self._check_contains(name, filter_params):
                try:
                    min_len = min([len(self.experiments[i].values[value]) for i in group.index])
                    max_len = max([len(self.experiments[i].values[value]) for i in group.index])
                    if use_min_len:
                        stacked_time_steps = np.stack(apply_func(self.experiments[i].values[value][:min_len]) for i in group.index)
                    else:
                        stacked_time_steps = np.stack(apply_func(self.experiments[i].values[value]) for i in group.index if len(self.experiments[i].values[value]) == max_len)
                    #TODO fix this, this is because they're stored wrong
                    #steps = np.stack([es.experiments[i].steps_of_values[value][0] for i in group.index])
                    if verbose:
                        print(f"Experiments {group.index} has {len(stacked_time_steps)} runs")
                    mu = np.mean(stacked_time_steps, axis=0).flatten()
                    var = np.var(stacked_time_steps, axis=0).flatten()
                    res[name] = {"mu":mu, "var":var}
                except Exception as e:
                    print(f"Experiments {group.index} exception..")
                    print(e)

        return res

import glob

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

warnings.simplefilter("ignore")
fxn()

class Experiment:
    def __init__(self, root_path, load_tensorboard, load_h5, load_metrics=False) -> None:
        self.root_path = root_path
        if load_metrics:
            self._load_metrics()
        self._load_settings()    
        if load_tensorboard: 
            self._accumulate_events()
        if load_h5:
            self._load_h5()

    def _hdf5_to_dict(self, dataset):
        d = {}
        for k in dataset:
            if hasattr(dataset[k], 'keys'):
                dd = self._hdf5_to_dict(dataset[k])
                for k1,v in dd.items():
                    d[k+'/'+k1] = np.array(v)
            else:
                d[k] = dataset[k]
        return d

    def _load_h5(self):
        for f in glob.glob(self.root_path+"/*/*.h5"):
            dataset = h5py.File(f, 'r')
            d = self._hdf5_to_dict(dataset)
            self.values = {k:v['value'] for k,v in d.items()}
            self.timesteps = {k:v['step'] for k,v in d.items()}
            return

    
    @staticmethod
    def is_experiment(root_path):
        files = os.listdir(root_path)
        # return ('metrics.csv' in files) and ('settings.json' in files)
        is_experiment =  ('settings.json' in files)
        if not is_experiment:
            print("Not an experiment", root_path)
        return is_experiment
    
    def _accumulate_events(self):
        #print("Checking", self.root_path)
        for root, dirs, files in os.walk(self.root_path):
            #print("Checking", root)
            summary_iterators = [EventAccumulator(os.path.join(root, dname)).Reload() for dname in filter(lambda x: 'events' in x, dirs)]
            if len(summary_iterators) == 0: continue
            tags = summary_iterators[0].Tags()['scalars']

            for it in summary_iterators:
                assert it.Tags()['scalars'] == tags

            out = defaultdict(list)
            steps_of_values = defaultdict(list)
            for tag in tags:
                steps = [e.step for e in summary_iterators[0].Scalars(tag)]

                for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                    assert len(set(e.step for e in events)) == 1

                    out[tag].append([e.value for e in events])
                    steps_of_values[tag].append(steps)
        self.values = out
        self.steps_of_values = steps_of_values

    def _load_settings(self):
        self.settings = smart_settings.load(os.path.join(self.root_path, 'settings.json'))
        self.settings_flat = fd(self.settings)
        
    def _load_metrics(self):
        self.metrics = pd.DataFrame.to_dict(pd.read_csv(os.path.join(self.root_path, 'metrics.csv')), orient='r')[0]

    @classmethod
    def extract_experiments(cls, paths, **kwargs):
        for path in paths:
            experiments = []
            for root in tqdm(glob.glob(path+"/working_directories/*")):
                try:
                    if Experiment.is_experiment(root): experiments.append(Experiment(root, **kwargs))
                except Exception as e:
                    print('Skipping', root, 'because of', e)
        return experiments