import click
import mlflow
import pandas as pd


@click.command()
@click.option("--disease-data", type=pd.DataFrame)
@click.option("--contact-transmissions", default=0.2, type=float)
@click.option("--contact-rate", default=10.0, type=float)
@click.option("--recovery-rate", default=1.0 / 14, type=float)
@click.option("--death-rate", default=0.05, type=float)
@click.option("--days-to-check", default=7, type=int)
def fit_sir_model(disease_data, contact_transmissions: float, contact_rate: float, recovery_rate: float,
                  death_rate: float, days_to_check: int):
    mlflow.log_param("contact_transmissions", contact_transmissions)
    mlflow.log_param("contact_rate", contact_rate)
    mlflow.log_param("recovery_rate", recovery_rate)
    mlflow.log_param("death_rate", death_rate)
    mlflow.log_param("days_to_check", days_to_check)

    model = SIRModel(contact_transmissions, contact_rate, recovery_rate, death_rate)
    current_status = disease_data.iloc[-1]
    future_predictions = model.predict(current_status["Susceptible"],
                                       current_status["Infected"],
                                       current_status["Recovered"],
                                       current_status["Dead"],
                                       days_to_check)
    future_predictions.set_index(future_predictions.index + current_status.name, inplace=True)
    return future_predictions


_DEFAULT_TIME_SCALE = 12 * 3 * 31  # 36 months


class SIRModel:
    def __init__(
            self,
            transmission_rate_per_contact,
            contact_rate,
            recovery_rate,
            death_rate
    ):
        """
        Assumes hospital beds are enough, no change in treatment.
        :param transmission_rate_per_contact: Prob of contact between infected and susceptible leading to infection.
        :param contact_rate: Mean number of daily contacts between an infected individual and susceptible people.
        :param recovery_rate: Rate of recovery of infected individuals.
        :param death_rate: Average death rate.

        """
        self._infection_rate = transmission_rate_per_contact * contact_rate
        self._recovery_rate = recovery_rate
        # Death rate is amortized over the recovery period
        # since the chances of dying per day are mortality rate / number of days with infection
        self._death_rate = death_rate * recovery_rate

    def predict(self, susceptible, infected, recovered, dead, num_days):
        """
        Run simulation.
        :param susceptible: Number of susceptible people in population.
        :param infected: Number of infected people in population.
        :param recovered: Number of recovered people in population.
        :param dead: Number of dead people in the population
        :param num_days: Number of days to forecast.
        :return: List of values for S, I, R over time steps
        """
        population = susceptible + infected + recovered + dead
        mlflow.log_metric("population", population)

        S = [int(susceptible)]
        I = [int(infected)]
        R = [int(recovered)]
        D = [int(dead)]

        for t in range(num_days):
            # beta*I*S/N
            new_infected = self._infection_rate * I[-1] * S[-1] / population
            # dS/dt = new infections
            susceptible_t = S[-1] - new_infected
            # dI/dt = new infections - % no longer sick
            infected_t = (
                    I[-1]
                    + new_infected
                    - (self._death_rate + self._recovery_rate) * I[-1]
            )
            # dR/dt = % recovered
            recovered_t = R[-1] + self._recovery_rate * I[-1]
            # dR/dt = % recovered
            dead_t = D[-1] + self._death_rate * I[-1]

            S.append(round(susceptible_t))
            I.append(round(infected_t))
            R.append(round(recovered_t))
            D.append(round(dead_t))

        mlflow.log_metric(f"Susceptible at t ({num_days} days)", S[-1])
        mlflow.log_metric(f"Infected at t ({num_days} days)", I[-1])
        mlflow.log_metric(f"Recovered at t ({num_days} days)", R[-1])
        mlflow.log_metric(f"Dead at t ({num_days} days)", D[-1])

        return pd.DataFrame([[S[ind], I[ind], R[ind], D[ind]] for ind in range(num_days)],
                            columns=["Susceptible", "Infected", "Recovered", "Dead"],
                            index=range(num_days))


if __name__ == '__main__':
    fit_sir_model()
