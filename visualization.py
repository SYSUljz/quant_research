# --*-- coding: utf-8 --*--
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import pandas as pd
from pathlib import Path

import qlib
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
from qlib.log import get_module_logger

logger = get_module_logger("ResultAnalysis")


def generate_report(
    recorder_id: str,
    experiment_name: str,
    output_dir: str = "report_results"   # ⚡️修改：输出目录
):
    """
    从 Qlib 实验记录中生成多个单独的分析图表文件。
    每个图表会保存为独立的 HTML 文件。

    Args:
        recorder_id (str): 需要加载的实验记录 ID。
        experiment_name (str): 实验名称。
        output_dir (str): 保存 HTML 文件的文件夹路径 (默认: 'report_results')。
    """
    try:
        # 初始化 Qlib
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
        logger.info(f"Qlib 已初始化，正在加载实验 '{experiment_name}' 的记录 '{recorder_id}'")

        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)

        # 加载记录产物
        logger.info("正在加载预测和回测结果...")
        pred_df = recorder.load_object("pred.pkl")
        label_df = recorder.load_object("label.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        if "label" not in label_df.columns:
            label_df.columns = ["label"]
        logger.info("数据加载完成。")

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    # 合并预测值和标签
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    pred_label.dropna(inplace=True)

    # 输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"结果将保存到目录: {output_path.resolve()}")

    # --- 生成并单独保存图表 ---
    logger.info("开始生成图表...")

    # 1️⃣ 绩效报告 (report_graph & cumulative_return_graph)
    report_figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
    report_file = output_path / "report_graph.html"
    report_figs[0].write_html(str(report_file))
    logger.info(f"绩效报告图 已保存: {report_file}")

    # 2️⃣ IC 分析 (score_ic_graph)
    ic_figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    ic_file = output_path / "score_ic_graph.html"
    ic_figs[0].write_html(str(ic_file))
    logger.info(f"IC 分析图 已保存: {ic_file}")

    # 3️⃣ 风险分析 (risk_analysis_graph)
    risk_figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    risk_file = output_path / "risk_analysis_graph.html"
    risk_figs[0].write_html(str(risk_file))
    logger.info(f"风险分析图 已保存: {risk_file}")

    # 4️⃣ 模型性能分析 (rank_label_graph)
    perf_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    perf_file = output_path / "rank_label_graph.html"
    perf_figs[0].write_html(str(perf_file))
    logger.info(f"模型性能图 已保存: {perf_file}")

    logger.info("✅ 所有图表已成功生成并分别保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Qlib 实验记录生成单独保存的图表报告。")
    parser.add_argument("--recorder_id", type=str, default="212790176964803794", help="实验记录 ID")
    parser.add_argument("--experiment_name", type=str, default="tutorial_exp", help="实验名称")
    parser.add_argument("--output_dir", type=str, default="report_results", help="输出文件夹 (默认: report_results)")
    args = parser.parse_args()

    generate_report(
        recorder_id=args.recorder_id,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir
    )
