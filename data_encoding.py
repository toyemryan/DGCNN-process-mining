from pandas import read_csv
import networkx as nx
from ast import literal_eval
from os.path import join
from os import makedirs
import json
from torch_geometric.data import Data
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc
from config import MAX_WORKERS, INPUT_PATH
from functools import partial
_shared_data = None


def _encode(case_ig):
    global _shared
    name, ohe_activity, path = (
        _shared.get('name'),
        _shared.get('ohe_activity'),
        _shared.get('path')
    )

    _, ig = case_ig

    prefix = Data()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True)]
    while i < len(all_activities) - 2:
        prefix.case_id = ig.case_id
        prefix.set = ig.set
        prefix.y = torch.tensor([ohe_activity[all_activities[i + 1].get('activity')]], dtype=torch.float)
        prefix.next_activity = all_activities[i + 1].get('activity')
        prefix.prefix_size = i + 1
        prefix.concurrent_events = all_activities[i].get('n_concurrent_events')

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True)
                                              if node <= i)))
        edges = list(ig.subgraph(act_ids).edges(data=False))

        activities_x = list(map(lambda x:
                                ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')),
                                activities))

        prefix.x = torch.tensor(activities_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edges, dtype=torch.long).T

        if i > 0:
            torch.save(prefix, join(path, f'{name}_{prefix.case_id}_{prefix.prefix_size}.pt'))

        i += 1


def encode_prefix_igs(igs, shared_items):
    f = partial(_encode)
    with ProcessPoolExecutor(max_workers=int(MAX_WORKERS/2), initializer=init_worker, initargs=(shared_items,)) as executor:
        for _ in tqdm(executor.map(f, igs), total=len(igs), desc='Encoding prefix-IGs'):
            pass

    with open(join(shared_items.get('path'), 'done'), 'w') as _:
        pass


def _read(case_id):
    global _shared
    nodes, igs = (
        _shared.get('nodes'),
        _shared.get('igs'),
    )

    ig = igs.loc[igs['case_id'] == case_id]
    g = nx.MultiDiGraph()
    g.case_id, g.set = case_id, ig['set'].drop_duplicates().tolist()[0]

    for event in ig.itertuples(index=False):
        if event.type == 'v':
            temporal_features = [event.norm_time, event.trace_time, event.prev_event_time]
            g.add_node(event.node1,
                       activity=event.activity,
                       temporal_features=f'{temporal_features}'
                       )

        elif event.type == 'e':
            g.add_edge(event.node1, event.node2)

    return case_id, g


def read_igs(cases, shared_items):
    f = partial(_read)
    igs = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(shared_items,)) as executor:
        for out in tqdm(executor.map(f, cases), total=len(cases), desc='Reading IGs'):
            if out is not None:
                igs.append(out)

    return igs


def get_ohe_encoding(name, items, attribute):
    try:
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'r') as f:
            ohe = json.load(f)

    except (Exception, ):
        ohe = {item:[1 if i == j else 0 for j in range(len(items))] for i, item in enumerate(items)}
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'w') as f:
            f.write(json.dumps(ohe, indent=1))

    return ohe


def init_worker(shared_data):
    global _shared
    _shared = shared_data


def main(name):
    print('Reading dataset')
    igs = read_csv(join(INPUT_PATH, f'{name}_processed.g'), header=0, sep=',')

    nodes = igs.loc[igs['type'] == 'v']
    case_ids = nodes['case_id'].drop_duplicates().tolist()

    ohe_activity = get_ohe_encoding(name, nodes['activity'].drop_duplicates().tolist(), 'activity')

    shared_items = {
        'name': name,
        'igs': igs,
        'nodes': nodes,
        'ohe_activity': ohe_activity,
    }
    igs = read_igs(case_ids, shared_items)

    del nodes, shared_items
    gc.collect()

    path = join('dataset', f'{name}_tensors')
    makedirs(path, exist_ok=True)

    shared_items = {
        'name': name,
        'ohe_activity': ohe_activity,
        'path': path
    }

    encode_prefix_igs(igs, shared_items)

    del shared_items, igs
    gc.collect()
    