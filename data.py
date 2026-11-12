"""
Dataset loaders and zero-day splits for NSL-KDD and UNSW-NB15.

Every split returns a `Split` with:
    X_train, y_train, fam_train   -> used to fit XGBoost and the anomaly detectors
    X_val,   y_val,   fam_val     -> used ONLY for early stopping and thresholds
    X_test,  y_test,  fam_test    -> used ONLY for final scoring
    zd_test                       -> boolean mask, True where a test row is a "zero-day"

Labels: y = 0 normal, 1 attack. `fam_*` holds the attack family ('normal' for benign rows).
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ------------------------------------------------------------------ NSL-KDD

NSL_COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
               'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'intrusion_type', 'difficulty']
NSL_CATEGORICAL = ['protocol_type', 'service', 'flag']

# same 5-class grouping as create_multi_KDD() in the original utility_funcs.py
NSL_FAMILY = {'normal': 'normal'}
for k in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'mailbomb']:
    NSL_FAMILY[k] = 'DoS'
for k in ['ipsweep', 'satan', 'nmap', 'portsweep', 'saint', 'mscan']:
    NSL_FAMILY[k] = 'Probe'
for k in ['ftp_write', 'guess_passwd', 'warezmaster', 'warezclient', 'imap', 'phf', 'spy', 'multihop', 'named', 'xlock', 'sendmail',
          'xsnoop', 'worm', 'snmpgetattack', 'snmpguess']:
    NSL_FAMILY[k] = 'R2L'
for k in ['rootkit', 'loadmodule', 'buffer_overflow', 'perl', 'xterm', 'ps', 'sqlattack', 'httptunnel']:
    NSL_FAMILY[k] = 'U2R'
NSL_FAMILIES = ['DoS', 'Probe', 'R2L', 'U2R']

# ------------------------------------------------------------------ UNSW-NB15

UNSW_CATEGORICAL = ['proto', 'service', 'state']
UNSW_FAMILIES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']


@dataclass
class Split:
    name: str
    feature_names: list
    X_train: np.ndarray
    y_train: np.ndarray
    fam_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    fam_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    fam_test: np.ndarray
    zd_test: np.ndarray
    categories: dict = None   # {column: [value for code 0, value for code 1, ...]}; code -1 = unseen in training
    attack_test: np.ndarray = None   # attack type of each test row (e.g. 'snmpgetattack'; equals the family for UNSW-NB15)


def _ordinal_encode(train_df, other_dfs, columns):
    """
    Map each categorical value to its order of appearance in train (as the original code does); unseen values -> -1.
    Returns (train_df, other_dfs, categories) with categories[col] = list of original values indexed by code.
    """
    categories = {}
    for col in columns:
        values = list(train_df[col].unique())
        mapping = {v: i for i, v in enumerate(values)}
        categories[col] = [str(v) for v in values]
        train_df[col] = train_df[col].map(mapping)
        for df in other_dfs:
            df[col] = df[col].map(mapping).fillna(-1)
    return train_df, other_dfs, categories


def load_nsl_kdd():
    """Return (train_df, test_df) with columns: features..., 'attack', 'family', 'y'."""
    def read(fname):
        df = pd.read_csv(os.path.join(DATA_DIR, 'nsl_kdd', fname), names=NSL_COLUMNS, header=None)
        df = df.drop(columns=['difficulty']).rename(columns={'intrusion_type': 'attack'})
        df['family'] = df['attack'].map(NSL_FAMILY)
        assert df['family'].notna().all(), "unmapped NSL-KDD attack label"
        df['y'] = (df['attack'] != 'normal').astype(int)
        return df
    return read('KDDTrain+.txt'), read('KDDTest+.txt')


def load_unsw_nb15():
    """
    Return (train_df, test_df) with columns: features..., 'attack', 'family', 'y'.

    NB: the CSV file names in the public release are swapped. The official training partition has 175,341 rows,
    which is the file called test.csv in our download; the official testing partition (82,332 rows) is train.csv.
    """
    def read(fname):
        df = pd.read_csv(os.path.join(DATA_DIR, 'unsw_nb15', fname), encoding='utf-8-sig')
        df = df.drop(columns=['id'])
        df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
        df['family'] = df['attack_cat'].replace({'Normal': 'normal', 'Backdoors': 'Backdoor'})
        df['attack'] = df['family']
        df['y'] = df['label'].astype(int)
        return df.drop(columns=['attack_cat', 'label'])
    train, test = read('test.csv'), read('train.csv')
    assert len(train) == 175341 and len(test) == 82332, "unexpected UNSW-NB15 partition sizes"
    return train, test


def make_split(dataset, protocol, family=None, seed=0, val_size=0.2, max_train=None):
    """
    dataset:  'nsl_kdd' | 'unsw_nb15'
    protocol: 'official' -> (NSL-KDD only) zero-days are attack types in KDDTest+ never seen in KDDTrain+, as in Barnard et al.
              'lofo'     -> leave-one-family-out: every row of `family` is removed from train/val and becomes the zero-day class in test
    max_train: optional cap on train+val rows (stratified subsample), used by the --quick smoke test
    """
    if dataset == 'nsl_kdd':
        train_df, test_df = load_nsl_kdd()
        categorical = NSL_CATEGORICAL
    elif dataset == 'unsw_nb15':
        train_df, test_df = load_unsw_nb15()
        categorical = UNSW_CATEGORICAL
    else:
        raise ValueError(dataset)

    if protocol == 'official':
        assert dataset == 'nsl_kdd', "the official zero-day protocol is only defined for NSL-KDD"
        zd_test = ~test_df['attack'].isin(train_df['attack'].unique()).values
        name = f'{dataset}/official'
    elif protocol == 'lofo':
        assert family is not None
        train_df = train_df[train_df['family'] != family].reset_index(drop=True)
        zd_test = (test_df['family'] == family).values
        name = f'{dataset}/lofo/{family}'
    else:
        raise ValueError(protocol)

    if max_train is not None and len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=seed).reset_index(drop=True)

    meta = ['attack', 'family', 'y']
    feature_names = [c for c in train_df.columns if c not in meta]
    train_df, (test_df,), categories = _ordinal_encode(train_df.copy(), [test_df.copy()], categorical)

    # stratify on attack type (falls back to family/label if a type is too rare to stratify)
    strat = train_df['attack'].where(train_df['attack'].map(train_df['attack'].value_counts()) >= 2, train_df['family'])
    strat = strat.where(strat.map(strat.value_counts()) >= 2, train_df['y'].astype(str))
    idx_tr, idx_val = train_test_split(np.arange(len(train_df)), test_size=val_size, random_state=seed, stratify=strat)
    assert len(np.intersect1d(idx_tr, idx_val)) == 0
    tr, val = train_df.iloc[idx_tr], train_df.iloc[idx_val]
    if protocol == 'lofo':
        assert (tr['family'] != family).all() and (val['family'] != family).all(), "held-out family leaked into train/val"

    def arr(df):
        return df[feature_names].to_numpy(dtype=np.float64)

    return Split(name=name, feature_names=feature_names,
                 X_train=arr(tr), y_train=tr['y'].to_numpy(), fam_train=tr['family'].to_numpy(),
                 X_val=arr(val), y_val=val['y'].to_numpy(), fam_val=val['family'].to_numpy(),
                 X_test=arr(test_df), y_test=test_df['y'].to_numpy(), fam_test=test_df['family'].to_numpy(),
                 zd_test=zd_test, categories=categories, attack_test=test_df['attack'].to_numpy())


def all_splits(datasets=('nsl_kdd', 'unsw_nb15')):
    """List of (dataset, protocol, family) triples that make up the full experiment grid."""
    out = []
    if 'nsl_kdd' in datasets:
        out.append(('nsl_kdd', 'official', None))
        out += [('nsl_kdd', 'lofo', f) for f in NSL_FAMILIES]
    if 'unsw_nb15' in datasets:
        out += [('unsw_nb15', 'lofo', f) for f in UNSW_FAMILIES]
    return out
