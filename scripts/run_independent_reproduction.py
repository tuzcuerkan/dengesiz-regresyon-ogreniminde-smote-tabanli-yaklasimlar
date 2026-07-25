from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Iterable
import hashlib
import json
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('THESIS_RUN_ROOT', REPOSITORY_ROOT / '.runs'))
CODE_ROOT = REPOSITORY_ROOT / 'src'
SOURCE_CODE = CODE_ROOT
WORK = RUN_ROOT / 'independent_reproduction'
CODE = WORK / 'code'
OUT = WORK / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

# Independent code copy: the run uses a fresh copy of the canonical package.
PACKAGE_COPY = CODE / 'imbalanced_regression'
PACKAGE_COPY.mkdir(parents=True, exist_ok=True)
for name in ['__init__.py', 'data_preparation.py', 'fold_pipeline.py']:
    shutil.copy2(SOURCE_CODE / 'imbalanced_regression' / name, PACKAGE_COPY / name)

sys.path.insert(0, str(CODE))
from imbalanced_regression.data_preparation import DatasetBundle, load_all_main, make_grouped_splits  # noqa: E402
from imbalanced_regression.fold_pipeline import run_fold_model  # noqa: E402

MODELS = ['ekk', 'rf', 'smote_r_rf', 'smogn_rf', 'relevance_weighted_rf']
MODEL_POS = {m: i + 1 for i, m in enumerate(MODELS)}
DATASETS = [
    'abalone', 'california_housing', 'concrete', 'wine_quality_red',
    'air_quality_no2', 'servo', 'tgss_bmi', 'tgss_income',
]
DATASET_POS = {k: i + 1 for i, k in enumerate(DATASETS)}
SPLIT_SEEDS = (42, 43)
RF_TREES = 100
RESULT_PATH = OUT / 'independent_reproduction_fold_results.csv'
LOG_PATH = OUT / 'run.log'
BASELINE_PATH = Path(os.environ.get('THESIS_MAIN_BASELINE', REPOSITORY_ROOT / 'outputs' / 'results' / 'main_grouped_repeated5fold_fold_results.csv'))


def log(message: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {message}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_hash(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


def model_seed(dataset_pos: int, repeat: int, fold: int, model_key: str) -> int:
    return int(5 * 10_000_000 + dataset_pos * 100_000 + repeat * 10_000 + fold * 100 + MODEL_POS[model_key])


def actual_overlap(bundle: DatasetBundle, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[int, int]:
    train_groups = set(bundle.groups.iloc[train_idx].astype(str))
    test_groups = set(bundle.groups.iloc[test_idx].astype(str))
    overlap = train_groups.intersection(test_groups)
    overlap_rows = int(bundle.groups.iloc[test_idx].astype(str).isin(overlap).sum())
    return len(overlap), overlap_rows


def checkpoint(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')


def run() -> pd.DataFrame:
    if RESULT_PATH.exists():
        existing = pd.read_csv(RESULT_PATH)
        rows = existing.to_dict('records')
    else:
        rows = []
    completed = {str(r.get('task_id')) for r in rows}
    bundles = load_all_main()
    for dataset_key in DATASETS:
        bundle = bundles[dataset_key]
        splits = make_grouped_splits(bundle, 5, SPLIT_SEEDS)
        log(f'START {dataset_key}: {len(splits) * len(MODELS)} tasks')
        for split in splits:
            repeat = int(split['repeat'])
            fold = int(split['fold'])
            split_seed = int(split['seed'])
            train_idx = np.asarray(split['train_index'], dtype=int)
            test_idx = np.asarray(split['test_index'], dtype=int)
            overlap_n, overlap_rows = actual_overlap(bundle, train_idx, test_idx)
            for model_key in MODELS:
                task_id = f'main_grouped_repeated5fold|main|{dataset_key}|r{repeat}|f{fold}|{model_key}'
                if task_id in completed:
                    continue
                seed = model_seed(DATASET_POS[dataset_key], repeat, fold, model_key)
                started = time.perf_counter()
                row = run_fold_model(
                    bundle, train_idx, test_idx, model_key,
                    random_state=seed, n_estimators=RF_TREES,
                )
                row.update({
                    'protocol': 'main_grouped_repeated5fold',
                    'scenario': 'main',
                    'dataset_name': bundle.display_name,
                    'repeat': repeat,
                    'fold': fold,
                    'split_seed': split_seed,
                    'model_seed': seed,
                    'rf_n_estimators': RF_TREES,
                    'train_index_sha256': split_hash(train_idx),
                    'test_index_sha256': split_hash(test_idx),
                    'actual_group_overlap_n': overlap_n,
                    'overlap_test_rows_n': overlap_rows,
                    'test_group_n': int(bundle.groups.iloc[test_idx].astype(str).nunique()),
                    'elapsed_seconds': round(time.perf_counter() - started, 6),
                    'task_id': task_id,
                })
                rows.append(row)
                completed.add(task_id)
                checkpoint(rows)
            log(f'DONE {dataset_key} repeat={repeat} fold={fold}; total={len(rows)}')
        log(f'END {dataset_key}')
    return pd.DataFrame(rows)


def canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {'elapsed_seconds'}
    cols = sorted(c for c in df.columns if c not in exclude)
    return df.sort_values('task_id').reset_index(drop=True)[cols]


def canonical_hash(df: pd.DataFrame) -> str:
    # Stable binary-aware hash for the selected values, independent of CSV formatting.
    work = canonical_frame(df)
    h = hashlib.sha256()
    for col in work.columns:
        h.update(col.encode('utf-8') + b'\0')
        s = work[col]
        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64)
            h.update(arr.tobytes())
        else:
            for value in s.fillna('<NA>').astype(str):
                h.update(value.encode('utf-8') + b'\0')
    return h.hexdigest()


def compare(independent: pd.DataFrame) -> dict[str, Any]:
    baseline = pd.read_csv(BASELINE_PATH)
    b = baseline.sort_values('task_id').reset_index(drop=True)
    i = independent.sort_values('task_id').reset_index(drop=True)
    errors: list[str] = []
    if len(i) != 400:
        errors.append(f'Independent result count is {len(i)}, expected 400.')
    if set(i.task_id) != set(b.task_id):
        errors.append('Task ID sets differ.')
    key_cols = [
        'task_id','train_index_sha256','test_index_sha256','relevance_train_hash',
        'model_seed','split_seed','train_n','test_n','synthetic_n',
        'invalid_nominal_total','invalid_binary_total','invalid_onehot_group_total',
        'ordinal_noninteger_total','out_of_bounds_total','actual_group_overlap_n',
    ]
    structural = []
    for col in key_cols:
        if col not in b or col not in i:
            continue
        match = b[col].fillna('<NA>').astype(str).eq(i[col].fillna('<NA>').astype(str))
        structural.append({'field': col, 'matching_rows': int(match.sum()), 'total_rows': int(len(match)), 'all_match': bool(match.all())})
        if not match.all():
            errors.append(f'Structural field differs: {col}')
    structural_df = pd.DataFrame(structural)
    structural_df.to_csv(OUT / 'structural_comparison.csv', index=False, encoding='utf-8-sig')

    metric_cols = ['rmse','mae','r2','wmse','sera','oiha','rare_ratio','rare_rmse','rare_mae','normal_rmse','normal_mae']
    metric_rows = []
    exact_all = True
    for col in metric_cols:
        bv = pd.to_numeric(b[col], errors='coerce').to_numpy(dtype=float)
        iv = pd.to_numeric(i[col], errors='coerce').to_numpy(dtype=float)
        diff = np.abs(bv - iv)
        exact = np.equal(bv, iv) | (np.isnan(bv) & np.isnan(iv))
        close12 = np.isclose(bv, iv, rtol=0.0, atol=1e-12, equal_nan=True)
        metric_rows.append({
            'metric': col,
            'exact_rows': int(exact.sum()),
            'within_1e_12_rows': int(close12.sum()),
            'total_rows': int(len(bv)),
            'max_abs_difference': float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
            'mean_abs_difference': float(np.nanmean(diff)) if np.isfinite(diff).any() else 0.0,
        })
        exact_all = exact_all and bool(exact.all())
        if not close12.all():
            errors.append(f'Metric differs beyond 1e-12: {col}')
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT / 'metric_comparison.csv', index=False, encoding='utf-8-sig')

    # Summary reconstruction and comparison.
    group_cols = ['protocol','scenario','dataset_key','dataset_name','model_key']
    summary_metrics = metric_cols
    agg = {'evaluations': ('task_id','count'), 'synthetic_total': ('synthetic_n','sum')}
    for m in summary_metrics:
        agg[f'{m}_mean'] = (m,'mean')
        agg[f'{m}_std'] = (m,'std')
    summary_i = i.groupby(group_cols, as_index=False).agg(**agg)
    summary_b = b.groupby(group_cols, as_index=False).agg(**agg)
    summary_i.to_csv(OUT / 'independent_reproduction_main_summary.csv', index=False, encoding='utf-8-sig')
    merged = summary_b.merge(summary_i, on=group_cols, suffixes=('_baseline','_independent'))
    sum_rows = []
    for col in [c for c in summary_b.columns if c not in group_cols]:
        a = pd.to_numeric(merged[f'{col}_baseline'], errors='coerce').to_numpy(dtype=float)
        z = pd.to_numeric(merged[f'{col}_independent'], errors='coerce').to_numpy(dtype=float)
        d = np.abs(a-z)
        sum_rows.append({'field': col, 'max_abs_difference': float(np.nanmax(d)) if np.isfinite(d).any() else 0.0, 'within_1e_12': bool(np.isclose(a,z,rtol=0,atol=1e-12,equal_nan=True).all())})
    pd.DataFrame(sum_rows).to_csv(OUT / 'summary_comparison.csv', index=False, encoding='utf-8-sig')

    result = {
        'status': 'PASS' if not errors else 'FAIL',
        'phase': '7-independent-reproduction',
        'evaluations': int(len(i)),
        'baseline_evaluations': int(len(b)),
        'task_ids_identical': bool(set(i.task_id) == set(b.task_id)),
        'structural_fields_all_match': bool(all(r['all_match'] for r in structural)),
        'metrics_exact_all': exact_all,
        'metrics_within_1e_12_all': bool(all(r['within_1e_12_rows'] == r['total_rows'] for r in metric_rows)),
        'baseline_canonical_sha256': canonical_hash(b),
        'independent_canonical_sha256': canonical_hash(i),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__,
        },
        'target_environment_requested': {
            'python': '3.11.9',
            'numpy': '1.26.4',
            'pandas': '1.5.3',
            'scipy': '1.11.4',
            'scikit_learn': '1.4.2',
        },
        'target_environment_executed': False,
        'target_environment_blocker': 'The isolated runtime has no external network access and no local Python 3.11.9 interpreter/cache.',
        'errors': errors,
    }
    return result


def main() -> None:
    started = time.perf_counter()
    log('INDEPENDENT REPRODUCTION RUN START')
    independent = run()
    independent.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    validation = compare(independent)
    validation['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    (OUT / 'independent_reproduction_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = {
        'purpose': 'Independent reconstruction of the 400-evaluation main experiment.',
        'code_copy': 'Fresh copy in independent_reproduction/code.',
        'checkpoint_reuse_from_main_and_data_sensitivity': False,
        'data_sources': 'Same frozen raw-data area and canonical data preparation rules.',
        'splitter': 'group-aware repeated 5-fold, seeds 42 and 43',
        'models': MODELS,
        'rf_n_estimators': RF_TREES,
        'comparison_tolerance': {'absolute': 1e-12, 'relative': 0.0},
    }
    (OUT / 'independent_reproduction_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    files = []
    for p in sorted([*CODE.glob('*.py'), *OUT.glob('*')]):
        if p.is_file() and p.name != 'independent_reproduction_file_manifest.json':
            files.append({'relative_path': str(p.relative_to(WORK)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)})
    (OUT / 'independent_reproduction_file_manifest.json').write_text(json.dumps({'files': files}, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"INDEPENDENT REPRODUCTION RUN END status={validation['status']} evaluations={validation['evaluations']} elapsed={validation['elapsed_seconds']}s")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation['status'] != 'PASS':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
