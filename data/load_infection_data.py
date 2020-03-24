import os
import shutil
from pathlib import Path

import click
import mlflow
import pandas as pd
import tempfile

SOUTH_KOREA_POPULATION = 51269185


def load_disease_data(disease_data_path, total_population=SOUTH_KOREA_POPULATION):
    base_data = pd.read_csv(disease_data_path)
    base_data.index.name = "time"
    # current infected = total infected - recovered - dead
    base_data["curr_infected"] = base_data["confirmed"] - base_data["released"] - base_data["deceased"]
    # current susceptible ~= total pop - confirmed
    base_data["curr_susceptible"] = total_population - base_data["confirmed"]
    base_data.rename(columns={"released": "Recovered", "deceased": "Dead",
                              "curr_infected": "Infected", "curr_susceptible": "Susceptible"},
                     inplace=True)

    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "infections.parquet")
    result = base_data[["Susceptible", "Infected", "Recovered", "Dead"]]
    result.to_parquet(file_path, index=True)
    mlflow.log_artifacts(tmpdir, "infections-data")
    shutil.rmtree(tmpdir, ignore_errors=True)

    return result


@click.command()
@click.option("--disease-data-path", type=Path)
@click.option("--total-population", default=SOUTH_KOREA_POPULATION, type=int)
def load_disease_data_command(disease_data_path, total_population):
    load_disease_data(disease_data_path, total_population)


if __name__ == '__main__':
    load_disease_data_command()
