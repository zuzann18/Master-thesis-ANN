import csv
from datetime import datetime

import click
import pandas as pd
from keras.callbacks import CSVLogger

from constants import RESULTS_PATH
from data_processing import load_training_test_data
from experimental_config import EXPERIMENTAL_CONFIG
from models import get_model


@click.command()
@click.option('--experiment_id', type=int, help='Experiment ID', required=True)
def main(experiment_id):
    """
    Example:
    python run_experiments.py --experiment_id 0
    """
    config = [c for c in EXPERIMENTAL_CONFIG if c['experiment_id'] == experiment_id][0]
    run_experiment(**config)


def run_experiment(
        experiment_id,
        model_name,
        epochs=10,
        batch_size=64,
        dropout_rate=0,
        learning_rate=0.0001,
        optimizer='adam',
        augmentation=False,
):
    train_images, test_images, train_labels, test_labels = load_training_test_data()

    model = get_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer=optimizer,
        augmentation=augmentation,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    history_csv_file = RESULTS_PATH / f"training_history_{experiment_id}_{timestamp}.csv"

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_images, test_labels),
        callbacks=[CSVLogger(history_csv_file)]
    )

    pd.DataFrame(history.history).to_csv(history_csv_file, index=False)

    run_details = {
        'experiment_id': experiment_id,
        'timestamp': timestamp,
        'model_name': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
        'augmentation': augmentation,
        'history_csv_file': history_csv_file,
    }
    result_path = RESULTS_PATH / f"runs_history.csv"

    with open(result_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=run_details.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(run_details)


if __name__ == '__main__':
    main()
