#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from configs.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.data.dataset import Dataset, DatasetH
from qlib.data import D
import pandas as pd
import yaml
from visualization import generate_report

from visualization import generate_report
if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    factor_formula = "Ref($close, -20)/$close - 1"
    #model = SingleFactorModel(factor_formula=factor_formula, quantile=0.2)
    #model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    with open("configs/factors/momentum_20d.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Extract factor_config
    factor_cfg = cfg["factor_config"]
    model = init_instance_by_config(factor_cfg)

    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp

    #####################
    #train
    ####################
    with R.start(experiment_name="momentum_20d"):
        #R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        # # 训练模型
        # model.fit(dataset)

        # # 保存训练好的模型
        # R.save_objects(**{"model.pkl": model})
        recorder = R.get_recorder()
        # 生成预测并记录
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
    generate_report(
        recorder_id=recorder.id,
        experiment_name="momentum_20d",
        output_dir="report_results"
    )
    #####################
    # prediction
    ####################
    # with R.start(experiment_name="workflow_alpha"):
    #     #R.log_params(**flatten_dict(CSI300_GBDT_TASK))

    #     # 读取之前保存的模型
    #     recorder = R.get_recorder(recorder_id="603f0fb600ee47afac2b9e31d72868ac")

    #     #model = recorder.load_object("model.pkl")

    #     # 生成预测
    #     # sr = SignalRecord(model, dataset, recorder)
    #     # sr.generate()

    #     # 分析 & 回测
    #     sar = SigAnaRecord(recorder)
    #     sar.generate()

    #     par = PortAnaRecord(recorder, port_analysis_config, "day")
    #     par.generate()

    ##################
    # 3.test
    ##################
    # with R.start(experiment_name="record_test_exp"):
    #     #R.log_params(**flatten_dict(CSI300_GBDT_TASK))

    #     # 读取之前保存的模型
    #     recorder = R.get_recorder(recorder_id="836bfc3556534b938e61fd960d290991")

    #     #model = recorder.load_object("model.pkl")

    #     # 生成预测
    #     # sr = SignalRecord(model, dataset, recorder)
    #     # sr.generate()

    #     # 分析 & 回测
    #     sar = SigAnaRecord(recorder)
    #     sar.generate()

    #     par = PortAnaRecord(recorder, port_analysis_config, "day")
    #     par.generate()
    
