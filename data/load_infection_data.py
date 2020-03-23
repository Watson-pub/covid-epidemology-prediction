import os
from pathlib import Path

import click
import mlflow
import pandas as pd
import tempfile

SOUTH_KOREA_POPULATION = 51269185


@click.command()
@click.option("--disease-data-path", type=Path)
@click.option("--total-population", default=SOUTH_KOREA_POPULATION, type=int)
def load_disease_data(disease_data_path, total_population):
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
    data_dir = os.path.join(tmpdir, "disease data")
    result = base_data[["Susceptible", "Infected", "Recovered", "Dead"]]
    result.to_parquet(data_dir, index=True)
    mlflow.log_artifacts(data_dir, "infections-data")
    os.rmdir(tmpdir)

    return result


if __name__ == '__main__':
    load_disease_data()
