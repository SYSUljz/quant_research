from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from loguru import logger
from tqdm import tqdm

from all_dumper import DumpDataAll


class DumpDataFix(DumpDataAll):
    def _dump_instruments(self):
        logger.info("start dump instruments......")
        if self.is_db_source:
            new_stocks_data = list(
                filter(
                    lambda df: str(df[self.symbol_field_name].iloc[0]).upper()
                    not in self._old_instruments,
                    self.data_groups,
                )
            )
            with tqdm(total=len(new_stocks_data)) as p_bar:
                for df_group in new_stocks_data:
                    _begin_time, _end_time = self._get_date(df_group, is_begin_end=True)
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(
                        _end_time, pd.Timestamp
                    ):
                        symbol = str(df_group[self.symbol_field_name].iloc[0]).upper()
                        _dt_map = self._old_instruments.setdefault(symbol, dict())
                        _dt_map[self.INSTRUMENTS_START_FIELD] = self._format_datetime(
                            _begin_time
                        )
                        _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(
                            _end_time
                        )
                    p_bar.update()
        else:
            _fun = partial(self._get_date, is_begin_end=True)
            new_stock_files = sorted(
                filter(
                    lambda x: self.get_symbol_from_file(x).upper()
                    not in self._old_instruments,
                    self.df_files,
                )
            )
            with tqdm(total=len(new_stock_files)) as p_bar:
                with ProcessPoolExecutor(max_workers=self.works) as execute:
                    for file_path, (_begin_time, _end_time) in zip(
                        new_stock_files, execute.map(_fun, new_stock_files)
                    ):
                        if isinstance(_begin_time, pd.Timestamp) and isinstance(
                            _end_time, pd.Timestamp
                        ):
                            symbol = self.get_symbol_from_file(file_path).upper()
                            _dt_map = self._old_instruments.setdefault(symbol, dict())
                            _dt_map[self.INSTRUMENTS_START_FIELD] = (
                                self._format_datetime(_begin_time)
                            )
                            _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(
                                _end_time
                            )
                        p_bar.update()
        _inst_df = pd.DataFrame.from_dict(self._old_instruments, orient="index")
        _inst_df.index.names = [self.symbol_field_name]
        self.save_instruments(_inst_df.reset_index())
        logger.info("end of instruments dump.\n")

    def dump(self):
        self._calendars_list = self._read_calendars(
            self._calendars_dir.joinpath(f"{self.freq}.txt")
        )
        self._old_instruments = (
            self._read_instruments(
                self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME)
            )
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )
        self._dump_instruments()
        self._dump_features()
