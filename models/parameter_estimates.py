from pathlib import Path

import click


@click.command()
@click.option("--disease-data")
def estimate_model_parameters(disease_data: Path):
    """
    Estimate:
    1. Probability of infection given contact
    2. number of contacts per day
    3. recovery rate for infected (chance per day to recover = 1/number of days to recovery)
    4. death rate
    Estimate R0 = Probability of infection X number of contacts per day X number of infection days
    R0 is the average number of people an infected person infects

    :param disease_data:
    :return:
    """
    pass

if __name__ == '__main__':
    estimate_model_parameters()