EXPERIMENTAL_CONFIG = [
    # test on small and big data
    {
        'experiment_id': 0,
        'model_name': "CNN1",

    },
    {
        'experiment_id': 1,
        'model_name': "DNN1",
    },
    # result big data is better, lets test new architecture
    {
        'experiment_id': 2,
        'model_name': "NN2L",
    },
    {
        'experiment_id': 3,
        'model_name': "NN3L",
    },
    # result: 2Layers is better

    # Small dataset experiment
    {
        'experiment_id': 4,
        'model_name': "SimpleCNN",
    },

    # Hugging Face dataset experiment
    # Example configuration for CNN
    {
        'experiment_id': 5,
        'model_name': "SimpleCNN",
        'learning_rate': 0.001
    }
    # Incremental layer addition
    # Hyperparameter tuning
    # Adding Dropout
    # Advanced Architecture
]
