from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Iterable
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('THESIS_RUN_ROOT', REPOSITORY_ROOT / '.runs'))
CODE_ROOT = REPOSITORY_ROOT / 'src'
WORK = RUN_ROOT / 'protocol_sensitivity_10fold'
OUT = WORK / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_ROOT))

from imbalanced_regression.data_preparation import DatasetBundle, load_all_main, make_grouped_splits  # noqa: E402
from imbalanced_regression.fold_pipeline import run_fold_model  # noqa: E402

MODELS = ['ekk', 'rf', 'smote_r_rf', 'smogn_rf', 'relevance_weighted_rf']
MODEL_POS = {m: i + 1 for i, m in enumerate(MODELS)}
DATASET_ORDER = [
    'abalone', 'california_housing', 'concrete', 'wine_quality_red',
    'air_quality_no2', 'servo', 'tgss_bmi', 'tgss_income',
]
DATASET_POS = {k: i + 1 for i, k in enumerate(DATASET_ORDER)}
SPLIT_SEED = 42
RF_TREES = 100
RESULT_PATH = OUT / 'grouped_10fold_fold_results.csv'
LOG_PATH = OUT / 'run.log'
MAIN5_PATH = Path(os.environ.get('THESIS_MAIN5_BASELINE', REPOSITORY_ROOT / 'outputs' / 'results' / 'main_grouped_repeated5fold_fold_results.csv'))


def log(message: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {message}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_hash(values: Iterable[int]) -> str:
    arr = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def model_seed(dataset_pos: int, fold: int, model_key: str) -> int:
    return int(7 * 10_000_000 + dataset_pos * 100_000 + fold * 100 + MODEL_POS[model_key])


def task_id(dataset_key: str, fold: int, model_key: str) -> str:
    return f'protocol_sensitivity_grouped10fold|{dataset_key}|r1|f{fold}|{model_key}'


def save(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        'rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_ratio',
        'rare_rmse', 'rare_mae', 'normal_rmse', 'normal_mae',
    ]
    agg: dict[str, tuple[str, str]] = {
        'evaluations': ('task_id', 'count'),
        'synthetic_total': ('synthetic_n', 'sum'),
        'elapsed_total': ('elapsed_seconds', 'sum'),
        'group_overlap_total': ('actual_group_overlap_n', 'sum'),
        'min_test_n': ('test_n', 'min'),
        'max_test_n': ('test_n', 'max'),
        'min_rare_count': ('rare_count', 'min'),
        'max_rare_count': ('rare_count', 'max'),
    }
    for m in metrics:
        agg[f'{m}_mean'] = (m, 'mean')
        agg[f'{m}_std'] = (m, 'std')
    out = df.groupby(['protocol', 'dataset_key', 'dataset_name', 'model_key'], as_index=False).agg(**agg)
    out.to_csv(OUT / 'protocol_sensitivity_10fold_summary.csv', index=False, encoding='utf-8-sig')
    return out


def rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    lower = ['rmse', 'mae', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    for metric in lower:
        ranked[f'{metric}_rank'] = ranked.groupby('dataset_key')[f'{metric}_mean'].rank(method='average', ascending=True)
    ranked['r2_rank'] = ranked.groupby('dataset_key')['r2_mean'].rank(method='average', ascending=False)
    ranked.to_csv(OUT / 'protocol_sensitivity_10fold_summary_ranked.csv', index=False, encoding='utf-8-sig')
    return ranked


def summarize_main5(main5: pd.DataFrame) -> pd.DataFrame:
    metrics = ['rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    return main5.groupby(['dataset_key', 'dataset_name', 'model_key'], as_index=False)[metrics].mean().rename(
        columns={m: f'{m}_mean' for m in metrics}
    )


def protocol_comparison(main5_summary: pd.DataFrame, ten_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ['rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    ten = ten_summary[['dataset_key', 'dataset_name', 'model_key'] + [f'{m}_mean' for m in metrics]].copy()
    merged = main5_summary.merge(ten, on=['dataset_key', 'dataset_name', 'model_key'], suffixes=('_5x2', '_10fold'))
    for m in metrics:
        a = merged[f'{m}_mean_5x2'].astype(float)
        b = merged[f'{m}_mean_10fold'].astype(float)
        merged[f'{m}_delta'] = b - a
        merged[f'{m}_relative_delta_pct'] = np.where(np.abs(a) > 1e-15, (b - a) / np.abs(a) * 100.0, np.nan)
    merged.to_csv(OUT / 'protocol_model_comparison.csv', index=False, encoding='utf-8-sig')

    winner_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    lower = {'rmse', 'mae', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse'}
    for dataset in DATASET_ORDER:
        sub = merged.loc[merged.dataset_key == dataset].set_index('model_key')
        for metric in metrics:
            a = sub[f'{metric}_mean_5x2']
            b = sub[f'{metric}_mean_10fold']
            ascending = metric in lower
            winner5 = a.idxmin() if ascending else a.idxmax()
            winner10 = b.idxmin() if ascending else b.idxmax()
            ranks5 = a.rank(method='average', ascending=ascending)
            ranks10 = b.rank(method='average', ascending=ascending)
            rho = float(spearmanr(ranks5, ranks10).statistic)
            winner_rows.append({
                'dataset_key': dataset,
                'metric': metric,
                'winner_5fold_2repeat': winner5,
                'winner_10fold': winner10,
                'winner_same': bool(winner5 == winner10),
                'winner_value_5fold_2repeat': float(a.loc[winner5]),
                'winner_value_10fold': float(b.loc[winner10]),
            })
            rank_rows.append({
                'dataset_key': dataset,
                'metric': metric,
                'spearman_rank_correlation': rho,
                'max_absolute_rank_change': float(np.max(np.abs(ranks10 - ranks5))),
                'mean_absolute_rank_change': float(np.mean(np.abs(ranks10 - ranks5))),
            })
    winners = pd.DataFrame(winner_rows)
    ranks = pd.DataFrame(rank_rows)
    winners.to_csv(OUT / 'protocol_winner_agreement.csv', index=False, encoding='utf-8-sig')
    ranks.to_csv(OUT / 'protocol_rank_stability.csv', index=False, encoding='utf-8-sig')
    return merged, winners, ranks


def validate(df: pd.DataFrame, summary: pd.DataFrame, winners: pd.DataFrame, ranks: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    def require(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)
    require(len(df) == 400, f'400 yerine {len(df)} değerlendirme var.')
    require(df.dataset_key.nunique() == 8, 'Sekiz veri seti yok.')
    require(df.model_key.nunique() == 5, 'Beş model yok.')
    require((df.groupby(['dataset_key', 'model_key']).size() == 10).all(), 'Her veri seti-model çiftinde 10 değerlendirme yok.')
    require((df.actual_group_overlap_n == 0).all(), 'Eğitim-test grup örtüşmesi var.')
    invalid_cols = ['invalid_nominal_total', 'invalid_binary_total', 'invalid_onehot_group_total', 'ordinal_noninteger_total', 'out_of_bounds_total']
    require(df[invalid_cols].sum().sum() == 0, 'Geçersiz sentetik değer var.')
    essential = ['rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    require(np.isfinite(df[essential].to_numpy(dtype=float)).all(), 'Sonlu olmayan temel metrik var.')
    require((df.groupby(['dataset_key', 'repeat', 'fold']).test_index_sha256.nunique() == 1).all(), 'Modeller aynı test indeksini kullanmıyor.')
    require((df.groupby(['dataset_key', 'repeat', 'fold']).relevance_train_hash.nunique() == 1).all(), 'Modeller aynı eğitim ilgililiğini kullanmıyor.')
    require((summary.evaluations == 10).all(), 'Özetlerde 10 değerlendirme yok.')
    require(len(winners) == 64, '64 veri seti-metrik kazanan karşılaştırması yok.')
    require(len(ranks) == 64, '64 sıra kararlılığı kaydı yok.')
    require((df.rare_count > 0).all(), 'En az bir test katında nadir gözlem yok.')
    return {
        'status': 'PASS' if not errors else 'FAIL',
        'phase': 'protocol_sensitivity_10fold',
        'evaluations': int(len(df)),
        'dataset_count': int(df.dataset_key.nunique()),
        'model_count': int(df.model_key.nunique()),
        'group_overlap_total': int(df.actual_group_overlap_n.sum()),
        'invalid_synthetic_value_total': int(df[invalid_cols].sum().sum()),
        'zero_rare_fold_model_count': int((df.rare_count == 0).sum()),
        'winner_agreement_count': int(winners.winner_same.sum()),
        'winner_comparisons': int(len(winners)),
        'winner_agreement_ratio': float(winners.winner_same.mean()),
        'mean_rank_correlation': float(ranks.spearman_rank_correlation.mean()),
        'median_rank_correlation': float(ranks.spearman_rank_correlation.median()),
        'errors': errors,
    }


def main() -> None:
    started_total = time.perf_counter()
    log('GROUPED 10-FOLD PROTOCOL SENSITIVITY START')
    bundles = load_all_main()
    if RESULT_PATH.exists():
        rows = pd.read_csv(RESULT_PATH).to_dict('records')
    else:
        rows = []
    completed = {str(r.get('task_id')) for r in rows}

    for dataset_key in DATASET_ORDER:
        bundle: DatasetBundle = bundles[dataset_key]
        splits = make_grouped_splits(bundle, 10, (SPLIT_SEED,))
        log(f'START {dataset_key}: {len(splits) * len(MODELS)} görev')
        for split in splits:
            fold = int(split['fold'])
            train_idx = np.asarray(split['train_index'], dtype=int)
            test_idx = np.asarray(split['test_index'], dtype=int)
            train_groups = set(bundle.groups.iloc[train_idx].astype(str))
            test_groups = set(bundle.groups.iloc[test_idx].astype(str))
            overlap = train_groups.intersection(test_groups)
            overlap_rows = int(bundle.groups.iloc[test_idx].astype(str).isin(overlap).sum())
            for model_key in MODELS:
                tid = task_id(dataset_key, fold, model_key)
                if tid in completed:
                    continue
                seed = model_seed(DATASET_POS[dataset_key], fold, model_key)
                started = time.perf_counter()
                row = run_fold_model(bundle, train_idx, test_idx, model_key, random_state=seed, n_estimators=RF_TREES)
                row.update({
                    'protocol': 'protocol_sensitivity_grouped10fold',
                    'scenario': 'main_10fold',
                    'dataset_name': bundle.display_name,
                    'repeat': 1,
                    'fold': fold,
                    'split_seed': SPLIT_SEED,
                    'model_seed': seed,
                    'rf_n_estimators': RF_TREES,
                    'train_index_sha256': index_hash(train_idx),
                    'test_index_sha256': index_hash(test_idx),
                    'actual_group_overlap_n': int(len(overlap)),
                    'overlap_test_rows_n': overlap_rows,
                    'test_group_n': int(len(test_groups)),
                    'elapsed_seconds': round(time.perf_counter() - started, 6),
                    'task_id': tid,
                })
                rows.append(row)
                completed.add(tid)
                save(rows)
            log(f'DONE {dataset_key} fold={fold}; toplam={len(rows)}')
        log(f'END {dataset_key}')

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    summary = summarize(df)
    rank_summary(summary)
    main5 = pd.read_csv(MAIN5_PATH)
    main5_summary = summarize_main5(main5)
    comparison, winners, ranks = protocol_comparison(main5_summary, summary)
    validation = validate(df, summary, winners, ranks)

    # Extra high-level summaries.
    metric_names = ['rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    effect_rows = []
    for metric in metric_names:
        rel = comparison[f'{metric}_relative_delta_pct'].replace([np.inf, -np.inf], np.nan)
        effect_rows.append({
            'metric': metric,
            'mean_signed_relative_delta_pct': float(rel.mean()),
            'mean_absolute_relative_delta_pct': float(rel.abs().mean()),
            'median_absolute_relative_delta_pct': float(rel.abs().median()),
            'max_absolute_relative_delta_pct': float(rel.abs().max()),
        })
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUT / 'protocol_effect_summary.csv', index=False, encoding='utf-8-sig')

    dataset_rows = []
    for dataset in DATASET_ORDER:
        w = winners[winners.dataset_key == dataset]
        r = ranks[ranks.dataset_key == dataset]
        ten_rows = df[df.dataset_key == dataset]
        dataset_rows.append({
            'dataset_key': dataset,
            'winner_agreement_count': int(w.winner_same.sum()),
            'winner_comparisons': int(len(w)),
            'winner_agreement_ratio': float(w.winner_same.mean()),
            'mean_rank_correlation': float(r.spearman_rank_correlation.mean()),
            'min_test_n': int(ten_rows.test_n.min()),
            'max_test_n': int(ten_rows.test_n.max()),
            'min_rare_count': int(ten_rows.rare_count.min()),
            'max_rare_count': int(ten_rows.rare_count.max()),
        })
    pd.DataFrame(dataset_rows).to_csv(OUT / 'protocol_dataset_stability.csv', index=False, encoding='utf-8-sig')

    validation.update({
        'elapsed_seconds': round(time.perf_counter() - started_total, 3),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'rf_parameters': {
            'n_estimators': RF_TREES,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
        },
    })
    (OUT / 'protocol_sensitivity_10fold_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = {
        'phase': 'protocol_sensitivity_10fold',
        'purpose': 'Test whether thesis conclusions depend materially on fold count.',
        'splitter': 'group-aware stratified group-level single 10-fold',
        'n_splits': 10,
        'n_repeats': 1,
        'split_seed': SPLIT_SEED,
        'datasets': DATASET_ORDER,
        'models': MODELS,
        'evaluations': 400,
        'comparison_protocol': 'group-aware repeated 5-fold with seeds 42 and 43',
        'preprocessing': 'identical to main protocol; fitted on training fold only',
        'resampling': 'identical mixed-type train-only SMOTE-R-like and SMOGN-like',
        'rf_n_estimators': RF_TREES,
    }
    (OUT / 'protocol_sensitivity_10fold_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')

    files = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = {'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files]}
    (OUT / 'protocol_sensitivity_10fold_file_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"GROUPED 10-FOLD PROTOCOL SENSITIVITY END status={validation['status']} evaluations={validation['evaluations']} elapsed={validation['elapsed_seconds']}s")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
