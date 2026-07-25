from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from .data_preparation import DatasetBundle
except ImportError:  # pragma: no cover
    from data_preparation import DatasetBundle


class ExperimentValidationError(RuntimeError):
    """Deney zinciri doğrulama hatası."""


def _array_hash(values: Iterable[Any]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _index_hash(index: pd.Index) -> str:
    text = "\x1f".join(map(str, index.tolist()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_mode(series: pd.Series, *, fallback: float = 0.0) -> float:
    nonmissing = pd.to_numeric(series, errors="coerce").dropna()
    if nonmissing.empty:
        return float(fallback)
    modes = nonmissing.mode(dropna=True)
    return float(modes.iloc[0] if not modes.empty else nonmissing.iloc[0])


@dataclass(frozen=True)
class RelevanceFunction:
    method: str
    threshold: float
    coef: float
    train_n: int
    train_y_hash_sha256: str
    q1: float
    median: float
    q3: float
    iqr: float
    lower_control: float
    upper_control: float
    y_min: float
    y_max: float
    degenerate: bool = False

    def transform(self, y: Iterable[Any]) -> np.ndarray:
        values = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ExperimentValidationError("İlgililik dönüşümü için sonlu ve boş olmayan hedef gerekir.")
        if self.degenerate:
            return np.zeros_like(values)
        phi = np.zeros_like(values)
        low = values < self.median
        high = values > self.median
        if self.median > self.lower_control:
            phi[low] = (self.median - values[low]) / (self.median - self.lower_control)
        if self.upper_control > self.median:
            phi[high] = (values[high] - self.median) / (self.upper_control - self.median)
        return np.clip(phi, 0.0, 1.0)

    def rare_mask(self, y: Iterable[Any]) -> np.ndarray:
        return self.transform(y) >= self.threshold

    def rare_boundary(self, tail: str) -> float:
        if tail == "low":
            return float(self.median - self.threshold * (self.median - self.lower_control))
        if tail == "high":
            return float(self.median + self.threshold * (self.upper_control - self.median))
        raise ValueError(tail)


def fit_iqr_relevance(y_train: Iterable[Any], threshold: float = 0.5, coef: float = 1.5) -> RelevanceFunction:
    values = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ExperimentValidationError("İlgililik fonksiyonu yalnız eksiksiz eğitim hedefiyle fit edilmelidir.")
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = float(q3 - q1)
    ymin, ymax = float(values.min()), float(values.max())
    degenerate = bool(np.isclose(iqr, 0.0) or np.isclose(ymin, ymax))
    lower = ymin if degenerate else float(q1 - coef * iqr)
    upper = ymax if degenerate else float(q3 + coef * iqr)
    if lower >= med:
        lower = ymin
    if upper <= med:
        upper = ymax
    return RelevanceFunction(
        method="iqr_two_tailed", threshold=float(threshold), coef=float(coef),
        train_n=int(values.size), train_y_hash_sha256=_array_hash(values),
        q1=float(q1), median=float(med), q3=float(q3), iqr=iqr,
        lower_control=float(lower), upper_control=float(upper),
        y_min=ymin, y_max=ymax, degenerate=degenerate,
    )


def assert_relevance_train_only(rel: RelevanceFunction, y_train: Iterable[Any]) -> None:
    values = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
    if rel.train_y_hash_sha256 != _array_hash(values):
        raise ExperimentValidationError("İlgililik fonksiyonu verilen eğitim hedefiyle fit edilmemiştir.")


@dataclass
class FoldPreprocessor:
    bundle: DatasetBundle
    scale_numeric: bool = True

    def __post_init__(self) -> None:
        self.continuous = list(self.bundle.continuous)
        self.ordinal = list(self.bundle.ordinal)
        self.nominal = list(self.bundle.nominal)
        self.binary = list(self.bundle.binary)
        declared = self.continuous + self.ordinal + self.nominal + self.binary
        missing = [c for c in declared if c not in self.bundle.X.columns]
        if missing:
            raise ExperimentValidationError(f"Şemada olup X içinde bulunmayan sütunlar: {missing}")
        undeclared = [c for c in self.bundle.X.columns if c not in declared]
        if undeclared:
            raise ExperimentValidationError(f"Veri türü tanımlanmamış sütunlar: {undeclared}")
        self.fixed_categories = self.bundle.fixed_categories or {}
        self.exclusive_binary_groups: list[tuple[str, ...]] = []
        prefix_groups: dict[str, list[str]] = {}
        for c in self.binary:
            prefix = c.split("_", 1)[0] if "_" in c else c
            prefix_groups.setdefault(prefix, []).append(c)
        for cols in prefix_groups.values():
            if len(cols) < 2:
                continue
            numeric = self.bundle.X[cols].apply(pd.to_numeric, errors="coerce")
            complete = numeric.notna().all(axis=1)
            if complete.any() and np.isclose(numeric.loc[complete].sum(axis=1), 1.0).all():
                self.exclusive_binary_groups.append(tuple(cols))
        grouped = {c for group in self.exclusive_binary_groups for c in group}
        self.independent_binary = [c for c in self.binary if c not in grouped]
        self.fitted = False

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        X = X_train.copy()
        self.fit_index_hash = _index_hash(X.index)
        self.fit_n = int(len(X))
        self.impute_values: dict[str, float] = {}
        self.numeric_mean: dict[str, float] = {}
        self.numeric_scale: dict[str, float] = {}
        self.numeric_bounds: dict[str, tuple[float, float]] = {}
        self.category_values: dict[str, tuple[float, ...]] = {}

        for c in self.continuous + self.ordinal:
            s = pd.to_numeric(X[c], errors="coerce")
            med = float(s.median())
            if not np.isfinite(med):
                raise ExperimentValidationError(f"{c}: eğitim katında medyan hesaplanamadı.")
            filled = s.fillna(med).astype(float)
            self.impute_values[c] = med
            self.numeric_mean[c] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.numeric_scale[c] = std if np.isfinite(std) and std > 0 else 1.0
            self.numeric_bounds[c] = (float(filled.min()), float(filled.max()))

        for c in self.nominal:
            fixed = tuple(float(v) for v in self.fixed_categories.get(c, ()))
            mode = _safe_mode(X[c], fallback=(fixed[0] if fixed else 0.0))
            if fixed and mode not in fixed:
                mode = fixed[0]
            self.impute_values[c] = mode
            observed = tuple(sorted(pd.to_numeric(X[c], errors="coerce").dropna().astype(float).unique()))
            cats = fixed or observed
            if not cats:
                cats = (mode,)
            self.category_values[c] = tuple(cats)

        for c in self.binary:
            mode = _safe_mode(X[c], fallback=0.0)
            mode = 1.0 if mode >= 0.5 else 0.0
            self.impute_values[c] = mode

        self.model_feature_names = self._make_feature_names(drop_first=True)
        self.distance_feature_names = self._make_feature_names(drop_first=False)
        self.fitted = True
        return self

    def _make_feature_names(self, *, drop_first: bool) -> list[str]:
        names = list(self.continuous + self.ordinal)
        for c in self.nominal:
            cats = self.category_values[c]
            use = cats[1:] if drop_first and len(cats) > 1 else cats
            names.extend([f"{c}__{v:g}" for v in use])
        names.extend(self.binary)
        return names

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise ExperimentValidationError("FoldPreprocessor fit edilmemiştir.")

    def transform_typed(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = pd.DataFrame(index=X.index)
        for c in self.continuous + self.ordinal + self.nominal + self.binary:
            out[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.impute_values[c]).astype(float)
        for c in self.nominal:
            cats = set(self.category_values[c])
            invalid = ~out[c].isin(cats)
            if invalid.any():
                out.loc[invalid, c] = self.impute_values[c]
        for c in self.binary:
            out[c] = np.where(out[c] >= 0.5, 1.0, 0.0)
        if out.isna().any().any() or not np.isfinite(out.to_numpy(dtype=float)).all():
            raise ExperimentValidationError("Dönüşüm sonrası eksik veya sonsuz değer kaldı.")
        return out

    def _encode(self, X_typed: pd.DataFrame, *, drop_first: bool, scale_numeric: bool) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        numeric = pd.DataFrame(index=X_typed.index)
        for c in self.continuous + self.ordinal:
            vals = X_typed[c].astype(float)
            if scale_numeric:
                vals = (vals - self.numeric_mean[c]) / self.numeric_scale[c]
            numeric[c] = vals
        if not numeric.empty:
            parts.append(numeric)
        for c in self.nominal:
            cats = self.category_values[c]
            use = cats[1:] if drop_first and len(cats) > 1 else cats
            onehot = pd.DataFrame(index=X_typed.index)
            for v in use:
                onehot[f"{c}__{v:g}"] = (X_typed[c].astype(float) == float(v)).astype(float)
            parts.append(onehot)
        if self.binary:
            parts.append(X_typed[self.binary].astype(float))
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X_typed.index)
        expected = self._make_feature_names(drop_first=drop_first)
        out = out.reindex(columns=expected, fill_value=0.0)
        if out.isna().any().any() or not np.isfinite(out.to_numpy(dtype=float)).all():
            raise ExperimentValidationError("Kodlanmış matriste eksik veya sonsuz değer var.")
        return out

    def transform_model(self, X: pd.DataFrame) -> pd.DataFrame:
        typed = self.transform_typed(X)
        return self._encode(typed, drop_first=True, scale_numeric=self.scale_numeric)

    def transform_distance(self, X: pd.DataFrame) -> pd.DataFrame:
        typed = self.transform_typed(X)
        return self._encode(typed, drop_first=False, scale_numeric=True)

    def model_from_typed(self, typed: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return self._encode(typed, drop_first=True, scale_numeric=self.scale_numeric)

    def distance_from_typed(self, typed: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return self._encode(typed, drop_first=False, scale_numeric=True)

    def validate_typed(self, typed: pd.DataFrame) -> dict[str, Any]:
        invalid_nominal: dict[str, int] = {}
        invalid_binary: dict[str, int] = {}
        invalid_onehot_groups: dict[str, int] = {}
        ordinal_noninteger: dict[str, int] = {}
        out_of_bounds: dict[str, int] = {}
        for c in self.nominal:
            invalid_nominal[c] = int((~typed[c].isin(self.category_values[c])).sum())
        for c in self.binary:
            invalid_binary[c] = int((~typed[c].isin([0.0, 1.0])).sum())
        for group in self.exclusive_binary_groups:
            label = "|".join(group)
            invalid_onehot_groups[label] = int((~np.isclose(typed[list(group)].sum(axis=1), 1.0)).sum())
        for c in self.ordinal:
            ordinal_noninteger[c] = int((~np.isclose(typed[c], np.rint(typed[c]))).sum())
        for c in self.continuous + self.ordinal:
            lo, hi = self.numeric_bounds[c]
            out_of_bounds[c] = int(((typed[c] < lo - 1e-12) | (typed[c] > hi + 1e-12)).sum())
        return {
            "invalid_nominal_total": int(sum(invalid_nominal.values())),
            "invalid_binary_total": int(sum(invalid_binary.values())),
            "invalid_onehot_group_total": int(sum(invalid_onehot_groups.values())),
            "ordinal_noninteger_total": int(sum(ordinal_noninteger.values())),
            "out_of_bounds_total": int(sum(out_of_bounds.values())),
            "invalid_nominal": invalid_nominal,
            "invalid_binary": invalid_binary,
            "invalid_onehot_groups": invalid_onehot_groups,
            "ordinal_noninteger": ordinal_noninteger,
            "out_of_bounds": out_of_bounds,
        }


@dataclass(frozen=True)
class ResamplingSummary:
    method: str
    original_n: int
    original_rare_count: int
    original_normal_count: int
    synthetic_n: int
    resampled_n: int
    final_rare_ratio: float
    k_neighbors: int
    random_state: int
    invalid_nominal_total: int
    invalid_binary_total: int
    invalid_onehot_group_total: int
    ordinal_noninteger_total: int
    out_of_bounds_total: int


@dataclass(frozen=True)
class ResamplingResult:
    X_typed: pd.DataFrame
    y: pd.Series
    summary: ResamplingSummary


def _tail_labels(y: np.ndarray, rel: RelevanceFunction) -> np.ndarray:
    return np.where(y < rel.median, "low", "high")


def _neighbor_pools(distance: np.ndarray, tails: np.ndarray, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pools: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for i in range(len(distance)):
        candidates = np.flatnonzero(tails == tails[i])
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            pools.append(np.asarray([i], dtype=int))
            dists.append(np.asarray([0.0], dtype=float))
            continue
        ds = np.linalg.norm(distance[candidates] - distance[i], axis=1)
        order = np.argsort(ds)[: min(k, len(candidates))]
        pools.append(candidates[order])
        dists.append(ds[order])
    return pools, dists


def mixed_resample_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: FoldPreprocessor,
    relevance: RelevanceFunction,
    *,
    method: str,
    k_neighbors: int = 5,
    target_rare_ratio: float = 0.5,
    noise_scale: float = 0.02,
    random_state: int = 42,
) -> ResamplingResult:
    if method not in {"none", "smote_r", "smogn"}:
        raise ExperimentValidationError(f"Bilinmeyen yeniden örnekleme yöntemi: {method}")
    assert_relevance_train_only(relevance, y_train)
    typed = preprocessor.transform_typed(X_train).reset_index(drop=True)
    y = pd.to_numeric(y_train, errors="coerce").reset_index(drop=True).astype(float)
    rare = relevance.rare_mask(y)
    n_rare, n_normal = int(rare.sum()), int((~rare).sum())
    if method == "none":
        validation = preprocessor.validate_typed(typed)
        return ResamplingResult(typed, y, ResamplingSummary(
            method="none", original_n=len(y), original_rare_count=n_rare,
            original_normal_count=n_normal, synthetic_n=0, resampled_n=len(y),
            final_rare_ratio=float(rare.mean()), k_neighbors=0, random_state=random_state,
            invalid_nominal_total=validation["invalid_nominal_total"],
            invalid_binary_total=validation["invalid_binary_total"],
            invalid_onehot_group_total=validation["invalid_onehot_group_total"],
            ordinal_noninteger_total=validation["ordinal_noninteger_total"],
            out_of_bounds_total=validation["out_of_bounds_total"],
        ))
    if n_rare < 1 or n_normal < 1:
        raise ExperimentValidationError(f"{method}: nadir ve normal eğitim gözlemleri gereklidir.")
    desired_rare = int(np.ceil(target_rare_ratio * n_normal / (1.0 - target_rare_ratio)))
    synthetic_n = max(0, desired_rare - n_rare)
    if synthetic_n == 0:
        validation = preprocessor.validate_typed(typed)
        return ResamplingResult(typed, y, ResamplingSummary(
            method=method, original_n=len(y), original_rare_count=n_rare,
            original_normal_count=n_normal, synthetic_n=0, resampled_n=len(y),
            final_rare_ratio=float(rare.mean()), k_neighbors=k_neighbors, random_state=random_state,
            invalid_nominal_total=validation["invalid_nominal_total"],
            invalid_binary_total=validation["invalid_binary_total"],
            invalid_onehot_group_total=validation["invalid_onehot_group_total"],
            ordinal_noninteger_total=validation["ordinal_noninteger_total"],
            out_of_bounds_total=validation["out_of_bounds_total"],
        ))

    rng = np.random.default_rng(random_state)
    rare_idx = np.flatnonzero(rare)
    rare_typed = typed.iloc[rare_idx].reset_index(drop=True)
    rare_y = y.iloc[rare_idx].to_numpy(dtype=float)
    distance = preprocessor.distance_from_typed(rare_typed).to_numpy(dtype=float)
    tails = _tail_labels(rare_y, relevance)
    pools, distances = _neighbor_pools(distance, tails, int(k_neighbors))
    all_finite_dist = np.concatenate([d[d > 0] for d in distances if np.any(d > 0)]) if any(np.any(d > 0) for d in distances) else np.asarray([0.0])
    distance_threshold = float(np.median(all_finite_dist))

    numeric_std = {c: float(typed[c].std(ddof=0)) for c in preprocessor.continuous + preprocessor.ordinal}
    y_std = float(y.std(ddof=0))
    synth_rows: list[dict[str, float]] = []
    synth_y: list[float] = []

    for _ in range(synthetic_n):
        i = int(rng.integers(0, len(rare_typed)))
        pool = pools[i]
        pos = int(rng.integers(0, len(pool)))
        j = int(pool[pos])
        chosen_dist = float(distances[i][pos])
        row_i = rare_typed.iloc[i]
        row_j = rare_typed.iloc[j]
        tail = tails[i]
        use_noise = method == "smogn" and chosen_dist > distance_threshold
        lam = float(rng.random())
        new: dict[str, float] = {}
        for c in preprocessor.continuous:
            if use_noise:
                val = float(row_i[c] + rng.normal(0.0, noise_scale * numeric_std[c]))
            else:
                val = float(row_i[c] + lam * (row_j[c] - row_i[c]))
            lo, hi = preprocessor.numeric_bounds[c]
            new[c] = float(np.clip(val, lo, hi))
        for c in preprocessor.ordinal:
            if use_noise:
                val = float(row_i[c] + rng.normal(0.0, noise_scale * numeric_std[c]))
            else:
                val = float(row_i[c] + lam * (row_j[c] - row_i[c]))
            lo, hi = preprocessor.numeric_bounds[c]
            new[c] = float(np.clip(np.rint(val), np.ceil(lo), np.floor(hi)))
        for c in preprocessor.nominal:
            new[c] = float(row_i[c] if rng.random() < 0.5 else row_j[c])
        for group in preprocessor.exclusive_binary_groups:
            source = row_i if rng.random() < 0.5 else row_j
            for c in group:
                new[c] = float(source[c])
        for c in preprocessor.independent_binary:
            new[c] = float(row_i[c] if rng.random() < 0.5 else row_j[c])

        if use_noise:
            ynew = float(rare_y[i] + rng.normal(0.0, noise_scale * y_std))
        else:
            ynew = float(rare_y[i] + lam * (rare_y[j] - rare_y[i]))
        if tail == "low":
            ynew = min(ynew, relevance.rare_boundary("low"))
        else:
            ynew = max(ynew, relevance.rare_boundary("high"))
        ynew = float(np.clip(ynew, relevance.y_min, relevance.y_max))
        # Sayısal yuvarlama eşik sınırında phi değerini 0,5'in çok az altına
        # düşürebilir. Böyle bir durumda hedef aynı kuyruğun gözlenen uç
        # değerine taşınarak nadir bölge üyeliği kesin olarak korunur.
        if not bool(relevance.rare_mask([ynew])[0]):
            ynew = relevance.y_min if tail == "low" else relevance.y_max
        synth_rows.append(new)
        synth_y.append(ynew)

    synth = pd.DataFrame(synth_rows, columns=typed.columns)
    validation = preprocessor.validate_typed(synth)
    if any(validation[k] for k in ["invalid_nominal_total", "invalid_binary_total", "invalid_onehot_group_total", "ordinal_noninteger_total", "out_of_bounds_total"]):
        raise ExperimentValidationError(f"Sentetik veri türü doğrulaması başarısız: {validation}")
    if not relevance.rare_mask(synth_y).all():
        raise ExperimentValidationError("Sentetik hedeflerin tamamı nadir bölgede kalmadı.")
    Xres = pd.concat([typed, synth], ignore_index=True)
    yres = pd.concat([y, pd.Series(synth_y, name=y.name)], ignore_index=True)
    final_ratio = float(relevance.rare_mask(yres).mean())
    if final_ratio + 1e-12 < target_rare_ratio:
        raise ExperimentValidationError(f"Hedef nadir oranına ulaşılamadı: {final_ratio}")
    return ResamplingResult(Xres, yres, ResamplingSummary(
        method=method, original_n=len(y), original_rare_count=n_rare,
        original_normal_count=n_normal, synthetic_n=synthetic_n,
        resampled_n=len(yres), final_rare_ratio=final_ratio,
        k_neighbors=int(k_neighbors), random_state=int(random_state),
        invalid_nominal_total=validation["invalid_nominal_total"],
        invalid_binary_total=validation["invalid_binary_total"],
        invalid_onehot_group_total=validation["invalid_onehot_group_total"],
        ordinal_noninteger_total=validation["ordinal_noninteger_total"],
        out_of_bounds_total=validation["out_of_bounds_total"],
    ))


def relevance_sample_weights(y_train: Iterable[Any], relevance: RelevanceFunction, min_weight: float = 0.05) -> np.ndarray:
    assert_relevance_train_only(relevance, y_train)
    weights = np.maximum(relevance.transform(y_train), float(min_weight))
    mean = float(weights.mean())
    return weights / mean if mean > 0 else np.ones_like(weights)


def make_model(model_key: str, *, random_state: int, n_estimators: int = 100) -> Any:
    if model_key == "ekk":
        return LinearRegression()
    if model_key in {"rf", "smote_r_rf", "smogn_rf", "relevance_weighted_rf"}:
        return RandomForestRegressor(
            n_estimators=int(n_estimators), max_depth=20, min_samples_split=2,
            min_samples_leaf=1, max_features="sqrt", bootstrap=True,
            n_jobs=-1, random_state=int(random_state),
        )
    raise ExperimentValidationError(f"Bilinmeyen model: {model_key}")


def _validate_arrays(y_true: Iterable[Any], y_pred: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    if yt.shape != yp.shape or yt.size == 0 or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ExperimentValidationError("Metrik dizileri geçersiz.")
    return yt, yp


def compute_metrics(y_true: Iterable[Any], y_pred: Iterable[Any], relevance: RelevanceFunction) -> dict[str, float | int | None]:
    yt, yp = _validate_arrays(y_true, y_pred)
    err = yt - yp
    se = err ** 2
    phi = relevance.transform(yt)
    phi_pred = relevance.transform(yp)
    weights_sum = float(phi.sum())
    wmse = float(np.sum(phi * se) / weights_sum) if weights_sum > 1e-12 else float(se.mean())
    thresholds = np.linspace(0.0, 1.0, 101)
    sera_curve = [float(se[phi >= t].sum() / len(yt)) if np.any(phi >= t) else 0.0 for t in thresholds]
    joint = np.maximum(phi, phi_pred)
    oiha_curve = [float(se[joint >= t].sum() / len(yt)) if np.any(joint >= t) else 0.0 for t in thresholds]
    rare = phi >= relevance.threshold
    normal = ~rare
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n": int(len(yt)),
        "rmse": float(np.sqrt(se.mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1.0 - se.sum() / ss_tot) if ss_tot > 1e-12 else None,
        "wmse": wmse,
        "sera": float(np.trapezoid(sera_curve, thresholds)),
        "oiha": float(np.trapezoid(oiha_curve, thresholds)),
        "rare_count": int(rare.sum()),
        "normal_count": int(normal.sum()),
        "rare_ratio": float(rare.mean()),
        "rare_rmse": float(np.sqrt(se[rare].mean())) if rare.any() else None,
        "rare_mae": float(np.abs(err[rare]).mean()) if rare.any() else None,
        "normal_rmse": float(np.sqrt(se[normal].mean())) if normal.any() else None,
        "normal_mae": float(np.abs(err[normal]).mean()) if normal.any() else None,
    }


def run_fold_model(
    bundle: DatasetBundle,
    train_index: np.ndarray,
    test_index: np.ndarray,
    model_key: str,
    *,
    random_state: int,
    n_estimators: int = 100,
) -> dict[str, Any]:
    train_index = np.asarray(train_index, dtype=int)
    test_index = np.asarray(test_index, dtype=int)
    if np.intersect1d(train_index, test_index).size:
        raise ExperimentValidationError("Train ve test satır indeksleri örtüşüyor.")
    train_groups = set(bundle.groups.iloc[train_index].astype(str))
    test_groups = set(bundle.groups.iloc[test_index].astype(str))
    if train_groups.intersection(test_groups):
        raise ExperimentValidationError("Train ve test grup kimlikleri örtüşüyor.")

    X_train = bundle.X.iloc[train_index].copy()
    X_test = bundle.X.iloc[test_index].copy()
    y_train = bundle.y.iloc[train_index].reset_index(drop=True).astype(float)
    y_test = bundle.y.iloc[test_index].reset_index(drop=True).astype(float)
    test_index_hash_before = _index_hash(X_test.index)

    prep = FoldPreprocessor(bundle).fit(X_train)
    rel = fit_iqr_relevance(y_train)
    assert_relevance_train_only(rel, y_train)

    resampling_method = "none"
    if model_key == "smote_r_rf":
        resampling_method = "smote_r"
    elif model_key == "smogn_rf":
        resampling_method = "smogn"

    res = mixed_resample_train_only(
        X_train, y_train, prep, rel, method=resampling_method,
        random_state=random_state, k_neighbors=5,
        target_rare_ratio=0.5, noise_scale=0.02,
    )
    X_fit = prep.model_from_typed(res.X_typed)
    X_eval = prep.transform_model(X_test)
    if list(X_fit.columns) != list(X_eval.columns):
        raise ExperimentValidationError("Eğitim ve test model sütunları eşleşmiyor.")
    if len(X_eval) != len(test_index) or _index_hash(X_test.index) != test_index_hash_before:
        raise ExperimentValidationError("Test verisinin satır yapısı değiştirilmiştir.")

    model = make_model(model_key, random_state=random_state, n_estimators=n_estimators)
    if model_key == "relevance_weighted_rf":
        weights = relevance_sample_weights(y_train, rel)
        model.fit(prep.transform_model(X_train), y_train, sample_weight=weights)
        weight_min, weight_mean, weight_max = float(weights.min()), float(weights.mean()), float(weights.max())
    else:
        model.fit(X_fit, res.y)
        weight_min = weight_mean = weight_max = None
    pred = model.predict(X_eval)
    metrics = compute_metrics(y_test, pred, rel)
    return {
        "dataset_key": bundle.key,
        "model_key": model_key,
        "train_n": int(len(train_index)),
        "test_n": int(len(test_index)),
        "model_feature_count": int(X_eval.shape[1]),
        "fit_index_hash": prep.fit_index_hash,
        "test_index_hash": test_index_hash_before,
        "relevance_train_hash": rel.train_y_hash_sha256,
        "resampling_method": res.summary.method,
        "synthetic_n": res.summary.synthetic_n,
        "resampled_train_n": res.summary.resampled_n,
        "final_rare_ratio_train": res.summary.final_rare_ratio,
        "invalid_nominal_total": res.summary.invalid_nominal_total,
        "invalid_binary_total": res.summary.invalid_binary_total,
        "invalid_onehot_group_total": res.summary.invalid_onehot_group_total,
        "ordinal_noninteger_total": res.summary.ordinal_noninteger_total,
        "out_of_bounds_total": res.summary.out_of_bounds_total,
        "sample_weight_min": weight_min,
        "sample_weight_mean": weight_mean,
        "sample_weight_max": weight_max,
        **metrics,
    }
