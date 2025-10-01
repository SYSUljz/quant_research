# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from workflow import ExperimentWorkflow


def main():
    """
    Main entry point for running a quantitative research experiment.

    This script initializes and runs an experiment based on a configuration file.
    Example usage:
        python runner.py --config_path ./configs/config.yaml
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run a Qlib experiment from a config file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment.",
    )
    args = parser.parse_args()

    # Initialize and run the workflow
    try:
        workflow = ExperimentWorkflow(config_path=args.config_path)
        workflow.run_experiment()
        workflow.generate_report()
        print(f"Experiment '{workflow.experiment_name}' completed successfully.")
        print(f"Report generated in '{workflow.report_output_dir}'.")
    except Exception as e:
        print(f"An error occurred during the workflow execution: {e}")


if __name__ == "__main__":
    main()
