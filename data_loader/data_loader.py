from pathlib import Path
from typing import Union

import pandas as pd


def read_as_df(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a csv or parquet file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file.
    **kwargs :
        Additional keyword arguments passed to the underlying pandas
        reader.
        if the file is a sqlite file, you should pass "table_name" to specify the query.

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path).expanduser()
    suffix = file_path.suffix.lower()

    keep_keys = {".csv": ("low_memory",)}
    kept_kwargs = {}
    for k in keep_keys.get(suffix, []):
        if k in kwargs:
            kept_kwargs[k] = kwargs[k]

    if suffix == ".csv":
        return pd.read_csv(file_path, **kept_kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path, **kept_kwargs)
    elif suffix == ".db":
        import sqlite3

        conn = sqlite3.connect(file_path)
        sql_query = kwargs.pop("sql_query", "SELECT * FROM stock_data")
        df = pd.read_sql_query(sql_query, conn, **kept_kwargs)
        conn.close()
        return df
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def fetch_from_sql(
    file_path: Union[str, Path], universe: list[str], table_name: str, **kwargs
) -> pd.DataFrame:
    """
    Read a sqlite file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the sqlite file.
    sql_query : str
        SQL query to execute.
    universe : list[str]
        List of instruments to include in the query.
    **kwargs :
        Additional keyword arguments passed to the underlying pandas
        reader.
    """
