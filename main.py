import config
from experiment_runner import ExperimentRunner as Exp

def main() -> None:
    engine = Exp.setup_database()
    exp_runner = Exp(engine)
    exp_runner.run()

if __name__ == "__main__":
    main()