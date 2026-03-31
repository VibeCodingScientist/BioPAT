"""Polars compatibility layer for old (0.12.x) and new (0.15+) API versions."""

import polars as pl

# Detect Polars version
_POLARS_VERSION = tuple(int(x) for x in pl.__version__.split('.')[:2])
_IS_OLD_POLARS = _POLARS_VERSION < (0, 14)


def group_by(df: pl.DataFrame, *args, **kwargs):
    """Compatibility wrapper for group_by/groupby."""
    if _IS_OLD_POLARS:
        return df.groupby(*args, **kwargs)
    return df.group_by(*args, **kwargs)


def unique(df: pl.DataFrame, subset=None, keep="first"):
    """Compatibility wrapper for unique/distinct."""
    if _IS_OLD_POLARS:
        if subset:
            return df.distinct(subset=subset, keep=keep)
        return df.distinct()
    if subset:
        return df.unique(subset=subset, keep=keep)
    return df.unique()


def n_unique(series: pl.Series):
    """Compatibility wrapper for n_unique."""
    if _IS_OLD_POLARS:
        return series.n_unique()
    return series.n_unique()


def is_in(col_expr, values):
    """Compatibility wrapper for is_in that handles sets."""
    # Old polars doesn't accept sets, convert to list
    if isinstance(values, set):
        values = list(values)
    return col_expr.is_in(values)


def sample_df(df: pl.DataFrame, n=None, fraction=None, seed=None, shuffle=False):
    """Compatibility wrapper for DataFrame.sample."""
    if _IS_OLD_POLARS:
        if fraction is not None:
            # Old polars uses frac parameter
            return df.sample(frac=fraction, seed=seed)
        return df.sample(n=n, seed=seed)
    if fraction is not None:
        return df.sample(fraction=fraction, seed=seed, shuffle=shuffle)
    return df.sample(n=n, seed=seed, shuffle=shuffle)


def df_height(df: pl.DataFrame) -> int:
    """Get DataFrame row count compatible with both versions."""
    if hasattr(df, 'height'):
        return df.height
    return len(df)


def iter_rows(df: pl.DataFrame, named=True):
    """Compatibility wrapper for iterating rows."""
    if _IS_OLD_POLARS:
        if named:
            return df.to_dicts()
        return df.rows()
    return df.iter_rows(named=named)


def to_list(series: pl.Series):
    """Compatibility wrapper for Series.to_list."""
    return series.to_list()


def str_len_chars(col_expr):
    """Compatibility wrapper for string length."""
    if _IS_OLD_POLARS:
        return col_expr.str.lengths()
    return col_expr.str.len_chars()


def fill_null(col_expr, value):
    """Compatibility wrapper for fill_null."""
    return col_expr.fill_null(value)


def concat(dfs, **kwargs):
    """Compatibility wrapper for concat."""
    if not dfs:
        return pl.DataFrame()
    return pl.concat(dfs, **kwargs)


def map_elements(col_expr, func, return_dtype=None):
    """Compatibility wrapper for map_elements/apply."""
    if _IS_OLD_POLARS:
        return col_expr.apply(func)
    return col_expr.map_elements(func, return_dtype=return_dtype)
