from pathlib import Path

import click
import mlflow

from data.load_infection_data import load_disease_data
from models.sir_model import fit_sir_model


@click.command()
@click.option("--disease-data-path", type=Path)
@click.option("--days-to-check", default=7, type=int)
def run_model(disease_data_path, days_to_check):
    with mlflow.start_run():
        disease_data = load_disease_data(disease_data_path)
        # contact_transmissions=42
        # contact_rate=42
        # recovery_rate=42
        # death_rate=42
        future_data = fit_sir_model(disease_data, days_to_check=days_to_check)


if __name__ == "__main__":
    run_model()
