from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import pandas as pd
from loguru import logger
from tqdm import tqdm

from base_dumper import DumpDataBase


class DumpDataAll(DumpDataBase):
    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set()
        date_range_list = []
        if self.is_db_source:
            logger.info("loading from db......")
            for df_group in tqdm(self.data_groups):
                (_begin_time, _end_time), _set_calendars = self._get_date(
                    df_group, as_set=True, is_begin_end=True
                )
                all_datetime.update(_set_calendars)
                if isinstance(_begin_time, pd.Timestamp) and isinstance(
                    _end_time, pd.Timestamp
                ):
                    _begin_time = self._format_datetime(_begin_time)
                    _end_time = self._format_datetime(_end_time)
                    symbol = str(df_group[self.symbol_field_name].iloc[0])
                    _inst_fields = [symbol.upper(), _begin_time, _end_time]
                    date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
        else:
            logger.info("loading from files......")
            _fun = partial(self._get_date, as_set=True, is_begin_end=True)
            with tqdm(total=len(self.df_files)) as p_bar:
                with ProcessPoolExecutor(max_workers=self.works) as executor:
                    for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                        self.df_files, executor.map(_fun, self.df_files)
                    ):
                        all_datetime.update(_set_calendars)
                        if isinstance(_begin_time, pd.Timestamp) and isinstance(
                            _end_time, pd.Timestamp
                        ):
                            _begin_time = self._format_datetime(_begin_time)
                            _end_time = self._format_datetime(_end_time)
                            symbol = self.get_symbol_from_file(file_path)
                            _inst_fields = [symbol.upper(), _begin_time, _end_time]
                            date_range_list.append(
                                f"{self.INSTRUMENTS_SEP.join(_inst_fields)}"
                            )
                        p_bar.update()

        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(
            map(pd.Timestamp, self._kwargs["all_datetime_set"])
        )
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")

    def _dump_features(self):
        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        iterable = self.data_groups if self.is_db_source else self.df_files
        with tqdm(total=len(iterable)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, iterable):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()
