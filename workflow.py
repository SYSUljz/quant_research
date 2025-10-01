# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

# Assuming visualization.py contains your report generation function
from visualization import generate_report


class ExperimentWorkflow:
    """
    Encapsulates the logic for setting up and running a Qlib experiment.
    """

    def __init__(self, config_path):
        """
        Initializes the workflow with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self._load_config(config_path)
        self.recorder = None

    def _load_config(self, config_path):
        """Loads the experiment configuration from a YAML file."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract key configurations for easy access
        self.qlib_init_config = self.config.get("qlib_init")
        self.task_config = self.config.get("task")
        self.port_analysis_config = self.config.get("port_analysis_config")
        self.experiment_name = self.config.get("experiment_name", "default_experiment")
        report_config = self.config.get("report_config", {})
        self.report_output_dir = report_config.get("output_dir", "report_results")

    def _setup_qlib(self):
        """Initializes Qlib and downloads data if necessary."""
        provider_uri = self.qlib_init_config.get("provider_uri")
        # Ensure data is available
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        # Initialize qlib
        qlib.init(**self.qlib_init_config)

    def _setup_components(self):
        """Initializes model and dataset from the configuration."""
        if not self.task_config:
            raise ValueError("Task configuration (model, dataset) is missing.")

        self.model = init_instance_by_config(self.task_config["model"])
        self.dataset = init_instance_by_config(self.task_config["dataset"])

        # You can add a sanity check for the dataset here
        print("Dataset and model initialized successfully.")
        example_df = self.dataset.prepare("train")
        print("Sample of prepared data:")
        print(example_df.head())

    def run_experiment(self):
        """
        Executes the main experiment logic, including model training,
        prediction, and backtesting.
        """
        self._setup_qlib()
        self._setup_components()

        with R.start(experiment_name=self.experiment_name):
            self.recorder = R.get_recorder()

            # Generate signals and save them
            sr = SignalRecord(self.model, self.dataset, self.recorder)
            sr.generate()

            # Run signal analysis
            sar = SigAnaRecord(self.recorder)
            sar.generate()

            # Run portfolio analysis (backtest)
            par = PortAnaRecord(self.recorder, self.port_analysis_config, "day")
            par.generate()

    def generate_report(self):
        """Generates a report for the completed experiment."""
        if not self.recorder:
            print("Recorder not found. Please run the experiment first.")
            return

        generate_report(
            recorder_id=self.recorder.id,
            experiment_name=self.experiment_name,
            output_dir=self.report_output_dir,
        )
