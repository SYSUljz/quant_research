import abc
import shutil
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from loguru import logger

from data_loader import read_as_df
from qlib.utils import fname_to_code, code_to_fname


class DumpDataBase(abc.ABC):
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        data_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
        table_name: str = None,
    ):
        data_path_obj = Path(data_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(
            filter(lambda x: len(x) > 0, map(str.strip, exclude_fields))
        )
        self._include_fields = tuple(
            filter(lambda x: len(x) > 0, map(str.strip, include_fields))
        )
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.table_name = table_name
        self.is_db_source = False
        self.data_groups = []

        if data_path_obj.suffix.lower() == ".db":
            self.is_db_source = True
            all_df = read_as_df(data_path_obj, table_name=self.table_name)
            if self.symbol_field_name not in all_df.columns:
                raise ValueError(
                    f"Symbol field '{self.symbol_field_name}' not found in the database table."
                )
            self.data_groups = [
                group for _, group in all_df.groupby(self.symbol_field_name)
            ]
            self.df_files = [data_path_obj]
            if limit_nums is not None:
                self.data_groups = self.data_groups[: int(limit_nums)]
        else:
            self.df_files = sorted(
                data_path_obj.glob(f"*{self.file_suffix}")
                if data_path_obj.is_dir()
                else [data_path_obj]
            )
            if limit_nums is not None:
                self.df_files = self.df_files[: int(limit_nums)]

        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = (
            backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        )
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = (
            self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT
        )

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self,
        file_or_df: [Path, pd.DataFrame],
        *,
        is_begin_end: bool = False,
        as_set: bool = False,
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = _calendars = pd.to_datetime(
                df[self.date_field_name].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = read_as_df(file_path, low_memory=False)
        if self.date_field_name in df.columns:
            df[self.date_field_name] = pd.to_datetime(df[self.date_field_name])
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.stem.strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else (
                set(df_columns) - set(self._exclude_fields)
                if self._exclude_fields
                else df_columns
            )
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )
        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(
            self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve()
        )
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(
            self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve()
        )
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[
                self.symbol_field_name
            ].apply(lambda x: fname_to_code(x.lower()).upper())
            instruments_data.to_csv(
                instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False
            )
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(
        self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]
    ) -> pd.DataFrame:
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype(
            "datetime64[ns]"
        )
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        return df.reindex(cal_df.index)

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(
        self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path
    ):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(
                f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}"
            )
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                np.hstack([date_index, _df[field]]).astype("<f").tofile(
                    str(bin_path.resolve())
                )

    def _dump_bin(
        self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]
    ):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(
                str(file_or_data.iloc[0][self.symbol_field_name]).lower()
            )
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return

        df = df.drop_duplicates(self.date_field_name)

        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()
