"""
tabpfn_robust_ensemble.py
─────────────────────────────────────────────────────────────────────────────
Fork of tabpfn-extensions/post_hoc_ensembles core.

Key additions vs. upstream AutoTabPFNRegressor:
  1. 完整预测矩阵记录  (predictions_matrix_: n_models × n_samples)
  2. Per-molecule 方差输出  (predict_variance)
  3. IQR 鲁棒加权推断  (iqr_weighted_predict)
  4. 无 AutoGluon 依赖，轻量可插拔

Designed for:
  EXP 3_4 pseudo_generation.py — 生成 pseudo_values.csv + pseudo_variances.csv

Usage:
    ens = TabPFNRobustEnsemble(n_models=20, task="regression", device="cpu")
    ens.fit(X_train, y_train)

    mean, var = ens.predict_mean_var(X_test)
    # or
    robust_mean = ens.iqr_weighted_predict(X_test)
    # or full matrix
    matrix = ens.predict_matrix(X_test)   # shape (n_models, n_test)

    ens.save_pseudo_csvs(X_test, mol_ids, prefix="ALOGP")
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ── TabPFN import (supports both tabpfn package and tabpfn-client) ──────────
try:
    from tabpfn import TabPFNRegressor, TabPFNClassifier
    _BACKEND = "tabpfn"
except ImportError:
    try:
        from tabpfn_client import TabPFNRegressor, TabPFNClassifier
        _BACKEND = "tabpfn_client"
    except ImportError as e:
        raise ImportError(
            "Neither 'tabpfn' nor 'tabpfn_client' is installed.\n"
            "  pip install tabpfn   (local, recommended for this fork)\n"
            "  pip install tabpfn-client  (cloud API)"
        ) from e


# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter space
# ══════════════════════════════════════════════════════════════════════════════

# Mirrors the search space used by upstream post_hoc_ensembles,
# trimmed to parameters accessible via the public TabPFN sklearn API.
_HP_SPACE_REGRESSION = {
    "n_estimators":        [4, 8, 16, 32],
    "subsample_samples":   [0.7, 0.8, 0.9, 1.0],
}

_HP_SPACE_CLASSIFICATION = {
    "n_estimators":        [4, 8, 16, 32],
    "subsample_samples":   [0.7, 0.8, 0.9, 1.0],
    "balance_probabilities": [True, False],
}


def _sample_configs(
    n: int,
    task: Literal["regression", "classification"],
    rng: np.random.Generator,
) -> list[dict]:
    """随机采样 n 个超参配置（含 random_state 隔离）。"""
    space = (
        _HP_SPACE_REGRESSION if task == "regression"
        else _HP_SPACE_CLASSIFICATION
    )
    configs = []
    for i in range(n):
        cfg = {k: rng.choice(v).item() for k, v in space.items()}
        cfg["random_state"] = int(rng.integers(0, 2**31))
        configs.append(cfg)
    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  IQR weighting
# ══════════════════════════════════════════════════════════════════════════════

def iqr_weighted_mean(
    matrix: np.ndarray,
    fence_scale: float = 1.5,
    fallback: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """
    Per-sample IQR 鲁棒加权均值。

    Parameters
    ----------
    matrix : np.ndarray, shape (n_models, n_samples)
        每个模型对每个样本的预测值。
    fence_scale : float
        IQR 围栏倍数，默认 1.5（Tukey fence）。
        设为 3.0 可只过滤极端离群。
    fallback : str
        当某样本所有模型均被判为离群时的回退策略。

    Returns
    -------
    np.ndarray, shape (n_samples,)
        IQR 过滤后的加权均值（每样本等权于非离群模型）。
    """
    n_models, n_samples = matrix.shape

    q1 = np.percentile(matrix, 25, axis=0)   # (n_samples,)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1

    lower = q1 - fence_scale * iqr
    upper = q3 + fence_scale * iqr

    # mask[i, j] = True  →  model i 对 sample j 的预测在围栏内
    mask = (matrix >= lower[None, :]) & (matrix <= upper[None, :])   # (n_models, n_samples)

    # 若某列全部被过滤 → fallback
    all_filtered = mask.sum(axis=0) == 0    # (n_samples,)

    weighted = np.where(mask, matrix, 0.0).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)

    if all_filtered.any():
        fb = (np.mean(matrix, axis=0) if fallback == "mean"
              else np.median(matrix, axis=0))
        weighted[all_filtered] = fb[all_filtered]

    return weighted


def iqr_filtered_variance(
    matrix: np.ndarray,
    fence_scale: float = 1.5,
) -> np.ndarray:
    """
    Per-sample 方差，仅在 IQR 围栏内的模型预测上计算。
    离群模型被排除后残差方差 = 真实不确定性的更干净估计。

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    n_models, n_samples = matrix.shape

    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - fence_scale * iqr
    upper = q3 + fence_scale * iqr

    mask = (matrix >= lower[None, :]) & (matrix <= upper[None, :])

    variances = np.full(n_samples, np.nan)
    for j in range(n_samples):
        inlier_preds = matrix[mask[:, j], j]
        if len(inlier_preds) >= 2:
            variances[j] = float(np.var(inlier_preds, ddof=1))
        elif len(inlier_preds) == 1:
            variances[j] = 0.0
        else:
            # 全部离群 → 回退到全局方差
            variances[j] = float(np.var(matrix[:, j], ddof=1))

    return variances


# ══════════════════════════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleRecord:
    """fit/predict 过程的完整记录，用于审计与调试。"""
    configs: list[dict] = field(default_factory=list)
    fit_sample_sizes: list[int] = field(default_factory=list)
    n_models_fitted: int = 0
    n_models_failed: int = 0
    failed_indices: list[int] = field(default_factory=list)


class TabPFNRobustEnsemble:
    """
    轻量 TabPFN 鲁棒集成器。

    Parameters
    ----------
    n_models : int
        集成的 TabPFN 子模型数量。建议 ≥ 15 使 IQR 统计稳定。
    task : "regression" | "classification"
    device : str
        传给 TabPFN 的 device 参数（"cpu", "cuda", "auto"）。
    fence_scale : float
        IQR 围栏倍数，控制离群过滤强度。
    random_state : int
        主随机种子，子模型种子由此派生。
    verbose : bool
        是否打印拟合进度。
    """

    def __init__(
        self,
        n_models: int = 20,
        task: Literal["regression", "classification"] = "regression",
        device: str = "cpu",
        fence_scale: float = 1.5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_models = n_models
        self.task = task
        self.device = device
        self.fence_scale = fence_scale
        self.random_state = random_state
        self.verbose = verbose

        # ── state after fit ──
        self.models_: list = []
        self.record_: EnsembleRecord = EnsembleRecord()
        self._is_fitted: bool = False

        # ── cached after predict ──
        self._last_matrix: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Fit
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: Optional[list[int]] = None,
    ) -> "TabPFNRobustEnsemble":
        """
        训练 n_models 个超参配置各异的 TabPFN。

        Parameters
        ----------
        X : array-like, shape (n_train, n_features)
        y : array-like, shape (n_train,)
        categorical_feature_indices : list[int] | None
            传给 TabPFN 的分类列索引（若后端支持）。
        """
        rng = np.random.default_rng(self.random_state)
        configs = _sample_configs(self.n_models, self.task, rng)

        self.models_ = []
        self.record_ = EnsembleRecord(configs=configs)

        ModelClass = TabPFNRegressor if self.task == "regression" else TabPFNClassifier

        for i, cfg in enumerate(configs):
            if self.verbose:
                print(f"  [TabPFNRobustEnsemble] fitting model {i+1}/{self.n_models}  cfg={cfg}")
            try:
                fit_kwargs: dict = {}
                if categorical_feature_indices is not None:
                    fit_kwargs["categorical_feature_indices"] = categorical_feature_indices

                model = ModelClass(device=self.device, **cfg)
                model.fit(X, y, **fit_kwargs)
                self.models_.append(model)
                self.record_.fit_sample_sizes.append(len(y))
                self.record_.n_models_fitted += 1
            except Exception as exc:
                warnings.warn(f"Model {i} failed to fit: {exc}")
                self.record_.n_models_failed += 1
                self.record_.failed_indices.append(i)

        if self.record_.n_models_fitted == 0:
            raise RuntimeError("All TabPFN sub-models failed to fit.")

        self._is_fitted = True
        if self.verbose:
            print(f"  [TabPFNRobustEnsemble] fitted {self.record_.n_models_fitted} models "
                  f"({self.record_.n_models_failed} failed)")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict: raw matrix
    # ─────────────────────────────────────────────────────────────────────────

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        返回完整预测矩阵。

        Returns
        -------
        np.ndarray, shape (n_fitted_models, n_samples)
            matrix[i, j] = 第 i 个子模型对第 j 个样本的预测。
        """
        self._check_fitted()
        preds = []
        for model in self.models_:
            if self.task == "regression":
                p = model.predict(X)
            else:
                p = model.predict_proba(X)   # (n_samples, n_classes)
            preds.append(p)

        matrix = np.stack(preds, axis=0)    # (n_models, n_samples) or (n_models, n_samples, n_classes)
        self._last_matrix = matrix
        return matrix

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict: mean & variance
    # ─────────────────────────────────────────────────────────────────────────

    def predict_mean_var(
        self,
        X: np.ndarray,
        use_iqr_filter: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        同时返回均值和方差（供 pseudo_generation 直接使用）。

        Parameters
        ----------
        use_iqr_filter : bool
            True  → IQR 鲁棒均值 + IQR 过滤后方差（推荐）
            False → 简单 mean + var（baseline）

        Returns
        -------
        mean : np.ndarray, shape (n_samples,)
        var  : np.ndarray, shape (n_samples,)
        """
        matrix = self.predict_matrix(X)   # (n_models, n_samples)

        if use_iqr_filter:
            mean = iqr_weighted_mean(matrix, fence_scale=self.fence_scale)
            var = iqr_filtered_variance(matrix, fence_scale=self.fence_scale)
        else:
            mean = matrix.mean(axis=0)
            var = matrix.var(axis=0, ddof=1)

        return mean, var

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict: sklearn-compatible single output
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, use_iqr_filter: bool = True) -> np.ndarray:
        """sklearn 兼容接口，返回 IQR 加权均值。"""
        mean, _ = self.predict_mean_var(X, use_iqr_filter=use_iqr_filter)
        return mean

    def iqr_weighted_predict(self, X: np.ndarray) -> np.ndarray:
        """语义化别名，等同于 predict(use_iqr_filter=True)。"""
        return self.predict(X, use_iqr_filter=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Variance statistics
    # ─────────────────────────────────────────────────────────────────────────

    def predict_variance(self, X: np.ndarray, use_iqr_filter: bool = True) -> np.ndarray:
        """仅返回方差，不返回均值。"""
        _, var = self.predict_mean_var(X, use_iqr_filter=use_iqr_filter)
        return var

    def variance_summary(self, X: np.ndarray) -> pd.DataFrame:
        """
        返回每个样本的方差诊断表，用于设置 drop_ratio 阈值。

        Columns: mean, var_iqr, var_naive, std_iqr, cv (变异系数), n_inliers
        """
        matrix = self.predict_matrix(X)
        mean_iqr = iqr_weighted_mean(matrix, self.fence_scale)
        var_iqr = iqr_filtered_variance(matrix, self.fence_scale)
        var_naive = matrix.var(axis=0, ddof=1)

        n_models = matrix.shape[0]
        q1 = np.percentile(matrix, 25, axis=0)
        q3 = np.percentile(matrix, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.fence_scale * iqr
        upper = q3 + self.fence_scale * iqr
        mask = (matrix >= lower[None, :]) & (matrix <= upper[None, :])
        n_inliers = mask.sum(axis=0)

        cv = np.where(
            np.abs(mean_iqr) > 1e-10,
            np.sqrt(var_iqr) / np.abs(mean_iqr),
            np.nan
        )

        return pd.DataFrame({
            "mean":      mean_iqr,
            "var_iqr":   var_iqr,
            "var_naive": var_naive,
            "std_iqr":   np.sqrt(var_iqr),
            "cv":        cv,
            "n_inliers": n_inliers,
        })

    # ─────────────────────────────────────────────────────────────────────────
    #  Export: pseudo_generation 专用
    # ─────────────────────────────────────────────────────────────────────────

    def save_pseudo_csvs(
        self,
        X_test: np.ndarray,
        mol_ids: list,
        prefix: str,
        output_dir: str = ".",
        use_iqr_filter: bool = True,
        also_save_matrix: bool = False,
    ) -> dict[str, str]:
        """
        一步生成 EXP 3_4 所需的两个 CSV 文件。

        Parameters
        ----------
        X_test : 测试集 2D 描述符
        mol_ids : 分子标识符列表（行索引）
        prefix : 端点名，例如 "ALOGP"
        output_dir : 输出目录
        use_iqr_filter : 是否使用 IQR 过滤
        also_save_matrix : 是否额外保存原始预测矩阵（供审计）

        Returns
        -------
        dict with keys: "values_path", "variances_path", ["matrix_path"]
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        mean, var = self.predict_mean_var(X_test, use_iqr_filter=use_iqr_filter)

        values_path = os.path.join(output_dir, f"{prefix}_pseudo_values.csv")
        variances_path = os.path.join(output_dir, f"{prefix}_pseudo_variances.csv")

        pd.DataFrame({"mol_id": mol_ids, f"{prefix}_pseudo": mean}).to_csv(
            values_path, index=False
        )
        pd.DataFrame({"mol_id": mol_ids, f"{prefix}_variance": var}).to_csv(
            variances_path, index=False
        )

        result = {"values_path": values_path, "variances_path": variances_path}

        if also_save_matrix and self._last_matrix is not None:
            matrix_path = os.path.join(output_dir, f"{prefix}_pred_matrix.csv")
            n_models = self._last_matrix.shape[0]
            df_matrix = pd.DataFrame(
                self._last_matrix.T,
                columns=[f"model_{i}" for i in range(n_models)],
            )
            df_matrix.insert(0, "mol_id", mol_ids)
            df_matrix.to_csv(matrix_path, index=False)
            result["matrix_path"] = matrix_path
            if self.verbose:
                print(f"  [save_pseudo_csvs] matrix saved → {matrix_path}")

        if self.verbose:
            print(f"  [save_pseudo_csvs] {prefix} → {values_path}")
            print(f"  [save_pseudo_csvs] {prefix} → {variances_path}")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before predicting.")

    def __repr__(self):
        status = f"fitted={self._is_fitted}"
        if self._is_fitted:
            status += f", n_fitted={self.record_.n_models_fitted}"
        return (
            f"TabPFNRobustEnsemble("
            f"n_models={self.n_models}, task={self.task!r}, "
            f"fence_scale={self.fence_scale}, {status})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience factory: drop-in for AutoTabPFNRegressor / AutoTabPFNClassifier
# ══════════════════════════════════════════════════════════════════════════════

def AutoTabPFNRobust(
    task: Literal["regression", "classification"] = "regression",
    n_models: int = 20,
    **kwargs,
) -> TabPFNRobustEnsemble:
    """工厂函数，语义上对应 upstream 的 AutoTabPFN{Regressor,Classifier}。"""
    return TabPFNRobustEnsemble(n_models=n_models, task=task, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  Quick usage demo (run as script)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    print("=" * 60)
    print("TabPFNRobustEnsemble — quick smoke test")
    print("=" * 60)

    X, y = make_regression(n_samples=300, n_features=20, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    ens = TabPFNRobustEnsemble(n_models=5, task="regression", device="cpu", verbose=True)
    ens.fit(X_train, y_train)

    # ── mean & var ──
    mean, var = ens.predict_mean_var(X_test)
    print(f"\nR² (IQR mean):  {r2_score(y_test, mean):.4f}")
    print(f"Variance range: [{var.min():.4f}, {var.max():.4f}]")

    # ── variance summary ──
    summary = ens.variance_summary(X_test)
    print(f"\nVariance summary (top 5 uncertain molecules):")
    print(summary.nlargest(5, "var_iqr").to_string())

    # ── export csvs ──
    mol_ids = [f"mol_{i}" for i in range(len(y_test))]
    paths = ens.save_pseudo_csvs(
        X_test, mol_ids, prefix="TARGET",
        output_dir="/tmp/pseudo_test",
        also_save_matrix=True,
    )
    print(f"\nSaved: {paths}")
    print("\n✓ All checks passed.")
