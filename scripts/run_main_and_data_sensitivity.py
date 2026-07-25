from __future__ import annotations

from dataclasses import replace
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
from sklearn.model_selection import KFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('THESIS_RUN_ROOT', REPOSITORY_ROOT / '.runs'))
CODE_ROOT = REPOSITORY_ROOT / 'src'
WORK = RUN_ROOT / 'main_and_data_sensitivity'
OUT = WORK / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_ROOT))

from imbalanced_regression.data_preparation import (  # noqa: E402
    DatasetBundle,
    load_all_main,
    load_bmi,
    load_air_quality,
    make_grouped_splits,
)
from imbalanced_regression.fold_pipeline import ExperimentValidationError, run_fold_model  # noqa: E402

MODELS = ['ekk', 'rf', 'smote_r_rf', 'smogn_rf', 'relevance_weighted_rf']
MODEL_POS = {m: i + 1 for i, m in enumerate(MODELS)}
MAIN_DATASET_ORDER = [
    'abalone', 'california_housing', 'concrete', 'wine_quality_red',
    'air_quality_no2', 'servo', 'tgss_bmi', 'tgss_income',
]
DATASET_POS = {k: i + 1 for i, k in enumerate(MAIN_DATASET_ORDER)}
BASE_SPLIT_SEEDS = (42, 43)
RF_TREES = 100

MAIN_PATH = OUT / 'main_fold_results.csv'
SENS_PATH = OUT / 'data_sensitivity_fold_results.csv'
LOG_PATH = OUT / 'run.log'


def log(message: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {message}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def task_id(protocol: str, scenario: str, dataset_key: str, repeat: int, fold: int, model_key: str) -> str:
    return f'{protocol}|{scenario}|{dataset_key}|r{repeat}|f{fold}|{model_key}'


def model_seed(protocol_code: int, dataset_pos: int, repeat: int, fold: int, model_key: str) -> int:
    # Stable, human-auditable seed formula. All values remain below 2^32.
    return int(protocol_code * 10_000_000 + dataset_pos * 100_000 + repeat * 10_000 + fold * 100 + MODEL_POS[model_key])


def split_index_hash(values: Iterable[int]) -> str:
    arr = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def actual_group_overlap(bundle: DatasetBundle, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[int, int, int]:
    tr = set(bundle.groups.iloc[train_idx].astype(str))
    te = set(bundle.groups.iloc[test_idx].astype(str))
    overlap = tr.intersection(te)
    overlap_rows = int(bundle.groups.iloc[test_idx].astype(str).isin(overlap).sum())
    return len(overlap), overlap_rows, len(te)


def enrich_row(
    row: dict[str, Any],
    *, protocol: str,
    scenario: str,
    display_name: str,
    repeat: int,
    fold: int,
    split_seed: int,
    model_seed_value: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_overlap_n: int,
    overlap_test_rows_n: int,
    test_group_n: int,
    elapsed: float,
) -> dict[str, Any]:
    row.update({
        'protocol': protocol,
        'scenario': scenario,
        'dataset_name': display_name,
        'repeat': int(repeat),
        'fold': int(fold),
        'split_seed': int(split_seed),
        'model_seed': int(model_seed_value),
        'rf_n_estimators': RF_TREES,
        'train_index_sha256': split_index_hash(train_idx),
        'test_index_sha256': split_index_hash(test_idx),
        'actual_group_overlap_n': int(group_overlap_n),
        'overlap_test_rows_n': int(overlap_test_rows_n),
        'test_group_n': int(test_group_n),
        'elapsed_seconds': round(float(elapsed), 6),
    })
    row['task_id'] = task_id(protocol, scenario, row['dataset_key'], repeat, fold, row['model_key'])
    return row


def execute_tasks(
    *,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: str,
    scenario: str,
    bundle: DatasetBundle,
    splits: list[dict[str, Any]],
    protocol_code: int,
    dataset_pos: int,
    overlap_reference_bundle: DatasetBundle | None = None,
) -> None:
    completed = {str(r.get('task_id')) for r in rows}
    start_count = len(rows)
    log(f'START {protocol}/{scenario}/{bundle.key}: {len(splits) * len(MODELS)} görev')
    for split in splits:
        repeat = int(split['repeat'])
        fold = int(split['fold'])
        split_seed = int(split['seed'])
        train_idx = np.asarray(split['train_index'], dtype=int)
        test_idx = np.asarray(split['test_index'], dtype=int)
        ref = overlap_reference_bundle or bundle
        group_overlap_n, overlap_test_rows_n, test_group_n = actual_group_overlap(ref, train_idx, test_idx)
        for model_key in MODELS:
            tid = task_id(protocol, scenario, bundle.key, repeat, fold, model_key)
            if tid in completed:
                continue
            seed = model_seed(protocol_code, dataset_pos, repeat, fold, model_key)
            started = time.perf_counter()
            row = run_fold_model(
                bundle, train_idx, test_idx, model_key,
                random_state=seed, n_estimators=RF_TREES,
            )
            elapsed = time.perf_counter() - started
            enrich_row(
                row,
                protocol=protocol,
                scenario=scenario,
                display_name=bundle.display_name,
                repeat=repeat,
                fold=fold,
                split_seed=split_seed,
                model_seed_value=seed,
                train_idx=train_idx,
                test_idx=test_idx,
                group_overlap_n=group_overlap_n,
                overlap_test_rows_n=overlap_test_rows_n,
                test_group_n=test_group_n,
                elapsed=elapsed,
            )
            rows.append(row)
            completed.add(tid)
            save_checkpoint(rows, checkpoint_path)
        log(f'DONE {protocol}/{scenario}/{bundle.key} repeat={repeat} fold={fold}; toplam satır={len(rows)}')
    log(f'END {protocol}/{scenario}/{bundle.key}: yeni {len(rows) - start_count} görev')


def row_level_kfold_splits(bundle: DatasetBundle) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    X_dummy = np.zeros((len(bundle.y), 1))
    for repeat, seed in enumerate(BASE_SPLIT_SEEDS, 1):
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_dummy), 1):
            results.append({
                'repeat': repeat,
                'fold': fold,
                'seed': seed,
                'train_index': train_idx,
                'test_index': test_idx,
            })
    return results


def summarize(results: pd.DataFrame, path: Path) -> pd.DataFrame:
    metrics = [
        'rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_ratio',
        'rare_rmse', 'rare_mae', 'normal_rmse', 'normal_mae',
    ]
    group_cols = ['protocol', 'scenario', 'dataset_key', 'dataset_name', 'model_key']
    agg: dict[str, tuple[str, str]] = {
        'evaluations': ('task_id', 'count'),
        'synthetic_total': ('synthetic_n', 'sum'),
        'elapsed_total': ('elapsed_seconds', 'sum'),
        'group_overlap_total': ('actual_group_overlap_n', 'sum'),
        'overlap_test_rows_total': ('overlap_test_rows_n', 'sum'),
    }
    for m in metrics:
        agg[f'{m}_mean'] = (m, 'mean')
        agg[f'{m}_std'] = (m, 'std')
    summary = results.groupby(group_cols, as_index=False).agg(**agg)
    summary.to_csv(path, index=False, encoding='utf-8-sig')
    return summary


def make_comparisons(main_summary: pd.DataFrame, sens_summary: pd.DataFrame) -> pd.DataFrame:
    pair_specs = [
        ('bmi_schema', 'tgss_bmi', 'tgss_bmi_workstat_sensitivity'),
        ('air_quality_scope', 'air_quality_no2', 'air_quality_no2_extended'),
        ('duplicate_split_policy_wine', 'wine_quality_red', 'wine_quality_red_standard_kfold'),
        ('duplicate_split_policy_concrete', 'concrete', 'concrete_standard_kfold'),
        ('duplicate_split_policy_income', 'tgss_income', 'tgss_income_standard_kfold'),
    ]
    rows: list[dict[str, Any]] = []
    metric_names = ['rmse', 'mae', 'r2', 'wmse', 'sera', 'oiha', 'rare_rmse', 'normal_rmse']
    for comparison, main_key, sens_key in pair_specs:
        left = main_summary.loc[main_summary.dataset_key == main_key].set_index('model_key')
        right = sens_summary.loc[sens_summary.dataset_key == sens_key].set_index('model_key')
        common = sorted(set(left.index).intersection(right.index))
        for model in common:
            row: dict[str, Any] = {
                'comparison': comparison,
                'main_dataset_key': main_key,
                'sensitivity_dataset_key': sens_key,
                'model_key': model,
            }
            for metric in metric_names:
                a = float(left.loc[model, f'{metric}_mean'])
                b = float(right.loc[model, f'{metric}_mean'])
                row[f'main_{metric}'] = a
                row[f'sensitivity_{metric}'] = b
                row[f'delta_{metric}'] = b - a
                row[f'relative_delta_{metric}_pct'] = ((b - a) / abs(a) * 100.0) if abs(a) > 1e-15 else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'main_and_data_sensitivity_sensitivity_comparisons.csv', index=False, encoding='utf-8-sig')
    return out


def add_ranks(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    lower = ['rmse_mean', 'mae_mean', 'wmse_mean', 'sera_mean', 'oiha_mean', 'rare_rmse_mean', 'normal_rmse_mean']
    ranked = summary.copy()
    for col in lower:
        ranked[col.replace('_mean', '_rank')] = ranked.groupby(['protocol', 'scenario', 'dataset_key'])[col].rank(method='average', ascending=True)
    ranked['r2_rank'] = ranked.groupby(['protocol', 'scenario', 'dataset_key'])['r2_mean'].rank(method='average', ascending=False)
    ranked.to_csv(path, index=False, encoding='utf-8-sig')
    return ranked


def validate_results(main: pd.DataFrame, sens: pd.DataFrame, main_summary: pd.DataFrame, sens_summary: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(len(main) == 400, f'Ana değerlendirme 400 değil: {len(main)}')
    require(len(sens) == 250, f'Duyarlılık değerlendirmesi 250 değil: {len(sens)}')
    require(main.dataset_key.nunique() == 8, f'Ana veri seti 8 değil: {main.dataset_key.nunique()}')
    require(main.model_key.nunique() == 5, f'Ana model 5 değil: {main.model_key.nunique()}')
    require((main.groupby(['dataset_key', 'model_key']).size() == 10).all(), 'Ana veri seti-model çiftlerinde 10 değerlendirme yok.')
    require((sens.groupby(['dataset_key', 'model_key']).size() == 10).all(), 'Duyarlılık veri seti-model çiftlerinde 10 değerlendirme yok.')
    require(len(main_summary) == 40, f'Ana özet satırı 40 değil: {len(main_summary)}')
    require(len(sens_summary) == 25, f'Duyarlılık özet satırı 25 değil: {len(sens_summary)}')

    essential = ['rmse', 'mae', 'wmse', 'sera', 'oiha']
    require(np.isfinite(main[essential].to_numpy(dtype=float)).all(), 'Ana sonuçlarda sonlu olmayan temel metrik var.')
    require(np.isfinite(sens[essential].to_numpy(dtype=float)).all(), 'Duyarlılık sonuçlarında sonlu olmayan temel metrik var.')
    require((main.actual_group_overlap_n == 0).all(), 'Ana protokolde grup örtüşmesi var.')
    require((sens.loc[~sens.scenario.str.contains('standard_kfold'), 'actual_group_overlap_n'] == 0).all(), 'Grup kontrollü duyarlılıkta örtüşme var.')
    invalid_cols = ['invalid_nominal_total', 'invalid_binary_total', 'invalid_onehot_group_total', 'ordinal_noninteger_total', 'out_of_bounds_total']
    require(main[invalid_cols].sum().sum() == 0, 'Ana sonuçlarda geçersiz sentetik değer var.')
    require(sens[invalid_cols].sum().sum() == 0, 'Duyarlılık sonuçlarında geçersiz sentetik değer var.')
    require((main.loc[main.model_key.isin(['smote_r_rf', 'smogn_rf']), 'synthetic_n'] > 0).all(), 'Ana yeniden örnekleme görevlerinden en az birinde sentetik üretim yok.')
    require((sens.loc[sens.model_key.isin(['smote_r_rf', 'smogn_rf']), 'synthetic_n'] > 0).all(), 'Duyarlılık yeniden örnekleme görevlerinden en az birinde sentetik üretim yok.')
    require(np.allclose(main.loc[main.model_key == 'relevance_weighted_rf', 'sample_weight_mean'], 1.0), 'Ana ağırlıklı RF ortalama ağırlığı 1 değil.')
    require(np.allclose(sens.loc[sens.model_key == 'relevance_weighted_rf', 'sample_weight_mean'], 1.0), 'Duyarlılık ağırlıklı RF ortalama ağırlığı 1 değil.')

    split_groups_main = main.groupby(['dataset_key', 'repeat', 'fold'])
    require((split_groups_main.test_index_sha256.nunique() == 1).all(), 'Ana modeller aynı test indeksini kullanmıyor.')
    require((split_groups_main.relevance_train_hash.nunique() == 1).all(), 'Ana modeller aynı eğitim ilgililiğini kullanmıyor.')
    split_groups_sens = sens.groupby(['dataset_key', 'repeat', 'fold'])
    require((split_groups_sens.test_index_sha256.nunique() == 1).all(), 'Duyarlılık modelleri aynı test indeksini kullanmıyor.')
    require((split_groups_sens.relevance_train_hash.nunique() == 1).all(), 'Duyarlılık modelleri aynı eğitim ilgililiğini kullanmıyor.')

    std = sens[sens.scenario.str.contains('standard_kfold')]
    if int(std.actual_group_overlap_n.sum()) == 0:
        warnings.append('Standart KFold duyarlılığında hiç grup örtüşmesi görülmedi; yinelenen kayıt etkisi sınırlı olabilir.')

    status = 'PASS' if not errors else 'FAIL'
    return {
        'status': status,
        'phase': '5-6',
        'main_evaluations': int(len(main)),
        'sensitivity_evaluations': int(len(sens)),
        'total_evaluations': int(len(main) + len(sens)),
        'main_dataset_count': int(main.dataset_key.nunique()),
        'main_model_count': int(main.model_key.nunique()),
        'main_group_overlap_total': int(main.actual_group_overlap_n.sum()),
        'standard_kfold_group_overlap_total': int(std.actual_group_overlap_n.sum()),
        'standard_kfold_overlap_test_rows_total': int(std.overlap_test_rows_n.sum()),
        'invalid_synthetic_value_total': int(main[invalid_cols].sum().sum() + sens[invalid_cols].sum().sum()),
        'errors': errors,
        'warnings': warnings,
    }


def main() -> None:
    total_started = time.perf_counter()
    log('MAIN AND DATA SENSITIVITY RUN START')
    bundles = load_all_main()
    main_rows = load_checkpoint(MAIN_PATH).to_dict('records')
    sens_rows = load_checkpoint(SENS_PATH).to_dict('records')

    main_splits: dict[str, list[dict[str, Any]]] = {}
    for key in MAIN_DATASET_ORDER:
        bundle = bundles[key]
        splits = make_grouped_splits(bundle, 5, BASE_SPLIT_SEEDS)
        main_splits[key] = splits
        execute_tasks(
            rows=main_rows,
            checkpoint_path=MAIN_PATH,
            protocol='main_grouped_repeated5fold',
            scenario='main',
            bundle=bundle,
            splits=splits,
            protocol_code=5,
            dataset_pos=DATASET_POS[key],
        )

    # Work-status sensitivity under the primary BMI row partitions.
    bmi_sensitivity = load_bmi(schema='workstat_sensitivity')
    if not np.allclose(bmi_sensitivity.y.to_numpy(), bundles['tgss_bmi'].y.to_numpy()):
        raise ExperimentValidationError('BMI sensitivity and primary targets are not aligned.')
    execute_tasks(
        rows=sens_rows,
        checkpoint_path=SENS_PATH,
        protocol='data_sensitivity_repeated5fold',
        scenario='bmi_workstat_schema_same_splits',
        bundle=bmi_sensitivity,
        splits=main_splits['tgss_bmi'],
        protocol_code=61,
        dataset_pos=DATASET_POS['tgss_bmi'],
        overlap_reference_bundle=bundles['tgss_bmi'],
    )

    # Air Quality extended scope under exactly the main scope row partitions.
    aq_extended = load_air_quality(scope='extended')
    if not np.allclose(aq_extended.y.to_numpy(), bundles['air_quality_no2'].y.to_numpy()):
        raise ExperimentValidationError('Air Quality main and extended targets are not aligned.')
    execute_tasks(
        rows=sens_rows,
        checkpoint_path=SENS_PATH,
        protocol='data_sensitivity_repeated5fold',
        scenario='air_quality_extended_scope_same_splits',
        bundle=aq_extended,
        splits=main_splits['air_quality_no2'],
        protocol_code=62,
        dataset_pos=DATASET_POS['air_quality_no2'],
        overlap_reference_bundle=bundles['air_quality_no2'],
    )

    # Row-level standard KFold duplicate sensitivity. The runner gets unique
    # synthetic group IDs so the deliberately non-grouped split can execute;
    # actual duplicate-group overlap is measured against the original bundle.
    for offset, key in enumerate(['wine_quality_red', 'concrete', 'tgss_income'], 1):
        original = bundles[key]
        unique_groups = pd.Series([f'row_{i}' for i in range(len(original.y))], name='row_group')
        runner_bundle = replace(
            original,
            key=f'{key}_standard_kfold',
            display_name=f'{original.display_name} standard KFold sensitivity',
            groups=unique_groups,
        )
        splits = row_level_kfold_splits(original)
        execute_tasks(
            rows=sens_rows,
            checkpoint_path=SENS_PATH,
            protocol='duplicate_sensitivity_repeated5fold',
            scenario='standard_kfold_row_level',
            bundle=runner_bundle,
            splits=splits,
            protocol_code=62 + offset,
            dataset_pos=DATASET_POS[key],
            overlap_reference_bundle=original,
        )

    main_df = pd.DataFrame(main_rows)
    sens_df = pd.DataFrame(sens_rows)
    main_df.to_csv(MAIN_PATH, index=False, encoding='utf-8-sig')
    sens_df.to_csv(SENS_PATH, index=False, encoding='utf-8-sig')

    main_summary = summarize(main_df, OUT / 'main_and_data_sensitivity_main_summary.csv')
    sens_summary = summarize(sens_df, OUT / 'main_and_data_sensitivity_sensitivity_summary.csv')
    main_ranked = add_ranks(main_summary, OUT / 'main_and_data_sensitivity_main_summary_ranked.csv')
    sens_ranked = add_ranks(sens_summary, OUT / 'main_and_data_sensitivity_sensitivity_summary_ranked.csv')
    comparisons = make_comparisons(main_summary, sens_summary)

    validation = validate_results(main_df, sens_df, main_summary, sens_summary)
    validation.update({
        'elapsed_seconds': round(time.perf_counter() - total_started, 3),
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
        'outputs': {
            'main_fold_results': MAIN_PATH.name,
            'sensitivity_fold_results': SENS_PATH.name,
            'main_summary': 'main_and_data_sensitivity_main_summary.csv',
            'sensitivity_summary': 'main_and_data_sensitivity_sensitivity_summary.csv',
            'sensitivity_comparisons': 'main_and_data_sensitivity_sensitivity_comparisons.csv',
        },
    })
    (OUT / 'main_and_data_sensitivity_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')

    protocol = {
        'phase': '5-6',
        'main_protocol': {
            'splitter': 'group-aware stratified group-level repeated 5-fold',
            'n_splits': 5,
            'n_repeats': 2,
            'split_seeds': list(BASE_SPLIT_SEEDS),
            'datasets': MAIN_DATASET_ORDER,
            'models': MODELS,
            'evaluations': 400,
        },
        'sensitivity_protocols': {
            'bmi_workstat_schema_same_splits': 50,
            'air_quality_extended_scope_same_splits': 50,
            'standard_kfold_row_level_wine_concrete_income': 150,
            'evaluations': 250,
        },
        'preprocessing': 'train-fold fit; median/mode imputation; numeric scaling; fixed-category encoding',
        'resampling': 'mixed-type train-only SMOTE-R-like and SMOGN-like; k=5; target rare ratio=0.5; no normal under-sampling',
        'relevance': 'train-only two-tailed IQR; coef=1.5; threshold=0.5',
        'metrics': ['RMSE', 'MAE', 'R2', 'WMSE', 'normalized SERA', 'normalized OİHA', 'rare/normal RMSE and MAE'],
        'rf_n_estimators': RF_TREES,
    }
    (OUT / 'main_and_data_sensitivity_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest_files = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = {
        'files': [
            {'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
            for p in manifest_files
        ]
    }
    (OUT / 'main_and_data_sensitivity_file_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    log(f"MAIN AND DATA SENSITIVITY RUN END status={validation['status']} total={validation['total_evaluations']} elapsed={validation['elapsed_seconds']}s")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
