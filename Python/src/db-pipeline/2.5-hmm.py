#!/usr/bin/env python3
"""
Pipeline Step 2.5: HMM Health State Detection

Unsupervised 4-state left-to-right HMM (Healthy -> Declining -> Critical -> Dead).
Two features per bin: MT activity (sum) and Pn position (variance).
No death labels used for training. Death boundary used only for evaluation.

Run AFTER:  1-prepare_data_and_health.py (and optionally 2-remove_flies.py)
Run BEFORE: 3-create_feature_table.py
"""

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from hmmlearn import hmm
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_CONFIG, DATABASE_URL

# ==================== CONSTANTS ====================
EXPERIMENT_ID = None        # None = auto-detect latest experiment
BIN_MINUTES = 5            # aggregate raw minute counts into bins
N_STATES = 4
DEATH_THRESHOLD_HOURS = 24  # for evaluation only (matches Step 1)
STATE_NAMES = ['Healthy', 'Declining', 'Critical', 'Dead']
STATE_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
N_PLOT = 5                  # example flies to plot
TRAIN_FRAC = 0.8            # fraction of flies used for training


# ==================== DATA LOADING ====================
def get_experiment_id():
    """Use EXPERIMENT_ID constant, or fall back to latest experiment."""
    if EXPERIMENT_ID is not None:
        return EXPERIMENT_ID
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1")
    eid = cur.fetchone()[0]
    conn.close()
    return eid


def load_all_readings(experiment_id):
    """Load MT and Pn readings. Returns dict: fly_id -> (mt_array, pn_array)."""
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT fly_id, reading_type, value FROM readings "
        "WHERE experiment_id = %(eid)s AND reading_type IN ('MT', 'Pn') "
        "ORDER BY fly_id, datetime",
        engine, params={'eid': experiment_id}
    )
    engine.dispose()
    result = {}
    for fid, grp in df.groupby('fly_id'):
        mt = grp[grp['reading_type'] == 'MT']['value'].values
        pn = grp[grp['reading_type'] == 'Pn']['value'].values
        if len(mt) > 0 and len(pn) > 0:
            n = min(len(mt), len(pn))
            result[fid] = (mt[:n], pn[:n])
    return result


# ==================== PREPROCESSING ====================
def bin_features(mt_raw, pn_raw, bin_min=BIN_MINUTES):
    """Bin into 2 features: MT sum and Pn variance per bin. Returns (n_bins, 2) array."""
    n = min(len(mt_raw), len(pn_raw)) // bin_min
    mt = mt_raw[:n * bin_min].reshape(n, bin_min).sum(axis=1).astype(float)
    pn = pn_raw[:n * bin_min].reshape(n, bin_min).astype(float)
    pn_var = np.var(pn, axis=1)
    return np.column_stack([mt, pn_var])


def find_death_boundary(raw, threshold_hours=DEATH_THRESHOLD_HOURS):
    """First minute index where threshold_hours of continuous zero activity begins.
    Returns None if the fly never died."""
    th = threshold_hours * 60
    if len(raw) < th:
        return None
    is_zero = (raw == 0).astype(np.int32)
    cs = np.cumsum(np.concatenate([[0], is_zero]))
    rolling_zeros = cs[th:] - cs[:-th]
    hits = np.where(rolling_zeros == th)[0]
    return int(hits[0]) if len(hits) > 0 else None


# ==================== HMM ====================
def train_hmm(features_list):
    """Unsupervised left-to-right 4-state GaussianHMM on 2 features (MT sum, Pn var).

    Emission means initialized from data quantiles. Transition matrix fixed.
    Baum-Welch learns only the emission parameters.
    """
    all_data = np.vstack(features_list)  # (total_bins, 2)
    mt_all, pn_all = all_data[:, 0], all_data[:, 1]

    # quantile-based init for each feature
    mt_q = np.percentile(mt_all, [75, 50, 25, 5])
    pn_q = np.percentile(pn_all, [75, 50, 25, 5])
    mt_var = max(float(np.var(mt_all)), 0.01)
    pn_var = max(float(np.var(pn_all)), 0.01)

    model = hmm.GaussianHMM(
        n_components=N_STATES, covariance_type='diag',
        n_iter=100, params='mc', init_params='', random_state=42, #mct
    )
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
    model.transmat_ = np.array([
        [0.95, 0.05, 0.00, 0.00],   # Healthy -> Healthy | Declining
        [0.00, 0.85, 0.15, 0.00],   # Declining -> Declining | Critical
        [0.00, 0.00, 0.90, 0.10],   # Critical -> Critical | Dead
        [0.00, 0.00, 0.00, 1.00],   # Dead (absorbing)
    ])
    model.means_ = np.array([
        [mt_q[0], pn_q[0]],   # Healthy: high MT, high Pn variance
        [mt_q[1], pn_q[1]],   # Declining
        [mt_q[2], pn_q[2]],   # Critical
        [mt_q[3], pn_q[3]],   # Dead: near-zero both
    ])
    model.covars_ = np.array([
        [mt_var, pn_var],
        [mt_var, pn_var],
        [mt_var, pn_var],
        [0.1,    0.1],
    ])

    print(f'  Init means:  MT={[f"{m:.1f}" for m in mt_q]}  Pn={[f"{m:.1f}" for m in pn_q]}')

    X = np.vstack(features_list)
    lengths = [len(f) for f in features_list]
    model.fit(X, lengths)

    print(f'  Trained means (MT, Pn_var):')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name:>10}: MT={model.means_[i,0]:.1f}  Pn={model.means_[i,1]:.2f}')
    return model


def predict_states(model, features):
    """Viterbi decode with post-hoc Dead-absorbing enforcement."""
    states = model.predict(features)
    first_dead = np.where(states == 3)[0]
    if len(first_dead) > 0:
        states[first_dead[0]:] = 3
    return states


# ==================== EVALUATION ====================
def evaluate(all_states, death_bins, fly_ids):
    """Return list of advance-warning hours for each dead fly."""
    advance = []
    for fid in fly_ids:
        db = death_bins.get(fid)
        if db is None:
            continue
        states = all_states[fid]
        warn = np.where((states == 2) | (states == 3))[0]
        if len(warn) > 0:
            advance.append((db - warn[0]) * BIN_MINUTES / 60)
    return advance


# ==================== VISUALIZATION ====================
def plot_fly(features, states, death_bin, fly_id, save_dir='hmm_plots'):
    """MT activity, Pn variance, and state assignments for one fly."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    hours = np.arange(len(features)) * BIN_MINUTES / 60

    for ax in (ax1, ax2):
        if death_bin is not None:
            ax.axvline(death_bin * BIN_MINUTES / 60, color='red', ls='--', alpha=0.5)

    ax1.plot(hours, features[:, 0], 'k-', alpha=0.7, lw=0.5)
    ax1.set_ylabel('MT activity')
    ax1.set_title(f'Fly {fly_id}')

    ax2.plot(hours, features[:, 1], 'k-', alpha=0.7, lw=0.5)
    ax2.set_ylabel('Pn variance')

    for i in range(len(states)):
        x0 = hours[i]
        x1 = hours[i + 1] if i + 1 < len(hours) else x0 + BIN_MINUTES / 60
        ax3.axvspan(x0, x1, color=STATE_COLORS[states[i]], alpha=0.7)
    ax3.set_yticks(range(N_STATES))
    ax3.set_yticklabels(STATE_NAMES)
    ax3.set_ylabel('State')
    ax3.set_xlabel('Hours')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fly_{fly_id.replace("/", "-")}_hmm.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==================== DATABASE ====================
def save_results(experiment_id, all_states):
    """Create hmm_states table (if needed) and bulk-insert all state sequences."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hmm_states (
            experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
            fly_id VARCHAR(50) NOT NULL,
            bin_index INT NOT NULL,
            bin_minutes INT NOT NULL,
            state INT NOT NULL CHECK (state BETWEEN 0 AND 3),
            PRIMARY KEY (experiment_id, fly_id, bin_index),
            FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
        )
    """)
    cur.execute("DELETE FROM hmm_states WHERE experiment_id = %s", [experiment_id])

    rows = []
    for fly_id, states in all_states.items():
        for i, s in enumerate(states):
            rows.append((experiment_id, fly_id, i, BIN_MINUTES, int(s)))
        if len(rows) >= 10_000:
            execute_values(cur, "INSERT INTO hmm_states VALUES %s", rows)
            rows = []
    if rows:
        execute_values(cur, "INSERT INTO hmm_states VALUES %s", rows)

    conn.commit()
    conn.close()


# ==================== MAIN ====================
if __name__ == '__main__':
    eid = get_experiment_id()
    print(f'Experiment: {eid}')

    # --- load ---
    print('Loading MT + Pn readings...')
    raw_data = load_all_readings(eid)
    print(f'  {len(raw_data)} flies loaded')

    # --- preprocess ---
    print('Binning features & detecting death boundaries...')
    features = {}
    death_bins = {}
    for fid, (mt, pn) in raw_data.items():
        if len(mt) < BIN_MINUTES:
            continue
        features[fid] = bin_features(mt, pn)
        db_raw = find_death_boundary(mt)
        death_bins[fid] = db_raw // BIN_MINUTES if db_raw is not None else None

    dead_ids = [f for f in features if death_bins[f] is not None]
    alive_ids = [f for f in features if death_bins[f] is None]
    print(f'  Dead (by {DEATH_THRESHOLD_HOURS}h rule): {len(dead_ids)}, Alive: {len(alive_ids)}')

    # --- train / test split (all flies for training, dead flies held out for eval) ---
    all_fly_ids = list(features.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_fly_ids)
    split = int(TRAIN_FRAC * len(all_fly_ids))
    train_ids = all_fly_ids[:split]
    held_out_ids = all_fly_ids[split:]
    held_out_dead = [f for f in held_out_ids if death_bins[f] is not None]

    print(f'\nTraining HMM (unsupervised) on {len(train_ids)} flies...')
    model = train_hmm([features[f] for f in train_ids])
    with open('hmm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('  Saved hmm_model.pkl')

    # --- predict all ---
    print('\nPredicting states for all flies...')
    all_states = {}
    for i, fid in enumerate(features):
        all_states[fid] = predict_states(model, features[fid])
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(features)}')
    print(f'  {len(all_states)} flies decoded')

    # --- evaluate against death boundary (held-out dead flies only) ---
    advance = evaluate(all_states, death_bins, held_out_dead)
    if advance:
        arr = np.array(advance)
        print(f'\nEvaluation on {len(held_out_dead)} held-out dead flies:')
        print(f'  (Death boundary = {DEATH_THRESHOLD_HOURS}h continuous zeros, used for eval only)')
        print(f'  Median advance warning: {np.median(arr):.1f}h')
        print(f'  Mean advance warning:   {np.mean(arr):.1f}h')
        print(f'  Predicted 6h+ early:    {np.sum(arr >= 6)}/{len(arr)} ({np.mean(arr >= 6):.0%})')

    # --- plot examples (mix of dead and alive) ---
    print(f'\nPlotting {N_PLOT} example flies...')
    plot_ids = (dead_ids[:N_PLOT] if dead_ids else list(features.keys())[:N_PLOT])
    for fid in plot_ids:
        plot_fly(features[fid], all_states[fid], death_bins.get(fid), fid)
        print(f'  Plotted {fid}')

    # --- save to DB ---
    print('\nSaving states to database...')
    save_results(eid, all_states)

    # --- summary ---
    all_s = np.concatenate(list(all_states.values()))
    print(f'\n{"=" * 50}')
    print(f'DONE - {len(all_states)} flies processed')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:>10}: {np.mean(all_s == i):6.1%} of time bins')
    print(f'  Output: hmm_states table, hmm_model.pkl, hmm_plots/')
