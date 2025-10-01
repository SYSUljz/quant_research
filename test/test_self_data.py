import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.constant import REG_CN
from pprint import pprint
from pathlib import Path
import pandas as pd

MARKET = "all"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"
provider_uri = (
    "/Users/jackli/Desktop/python/quant-research/.qlib/qlib_data/cn_data"  # target_dir
)
# GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)
df = D.features(
    ["000009.SZ"],
    ["$open", "$high", "$low", "$close", "$adj_factor"],
    start_time="2020-05-01",
    end_time="2020-05-31",
)
print(df)
