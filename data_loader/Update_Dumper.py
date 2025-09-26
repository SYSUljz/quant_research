import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from base_dumper import DumpDataBase
from data_loader import read_as_df
from qlib_utils import fname_to_code


class DumpDataUpdate(DumpDataBase):
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
    ):
        super().__init__(
            data_path,
            qlib_dir,
            backup_dir,
            freq,
            max_workers,
            date_field_name,
            file_suffix,
            symbol_field_name,
            exclude_fields,
            include_fields,
            limit_nums,
        )
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(
            self._calendars_dir.joinpath(f"{self.freq}.txt")
        )
        self._update_instruments = (
            self._read_instruments(
                self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME)
            )
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )

        self._all_data = self._load_all_source_data()
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(
                lambda x: x > self._old_calendar_list[-1],
                self._all_data[self.date_field_name].unique(),
            )
        )

    def _load_all_source_data(self):
        logger.info("start load all source data....")
        all_df = []

        def _read_df(file_path: Path):
            _df = read_as_df(file_path)
            if self.date_field_name in _df.columns and not np.issubdtype(
                _df[self.date_field_name].dtype, np.datetime64
            ):
                _df[self.date_field_name] = pd.to_datetime(_df[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for df in executor.map(_read_df, self.df_files):
                    if not df.empty:
                        all_df.append(df)
                    p_bar.update()

        logger.info("end of load all data.\n")
        return pd.concat(all_df, sort=False)

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(
                self.symbol_field_name, group_keys=False
            ):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (
                    isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)
                ):
                    continue
                if _code in self._update_instruments:
                    _update_calendars = (
                        _df[
                            _df[self.date_field_name]
                            > self._update_instruments[_code][
                                self.INSTRUMENTS_END_FIELD
                            ]
                        ][self.date_field_name]
                        .sort_values()
                        .to_list()
                    )
                    if _update_calendars:
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = (
                            self._format_datetime(_end)
                        )
                        futures[
                            executor.submit(self._dump_bin, _df, _update_calendars)
                        ] = _code
                else:
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(
                        _start
                    )
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[
                        executor.submit(self._dump_bin, _df, self.new_calendar_list)
                    ] = _code

            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"dump bin errors: {error_code}")

        logger.info("end of features dump.\n")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())
