# CIFAR-10 Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The project is structured into several Python scripts, each responsible for different parts of the workflow: data loading and preprocessing, model building, model training, model evaluation, and utility functions.

## Project Structure

- `data_loading_and_preprocessing.py`: Contains code for loading the CIFAR-10 dataset, preprocessing the data, and visualizing class distributions.
- `model_building.py`: Defines and compiles the CNN architecture.
- `model_training.py`: Contains code for training the CNN model with data augmentation and early stopping.
- `model_evaluation.py`: Evaluates the model's performance and visualizes predictions.
- `utility_functions.py`: Contains additional utility functions, such as creating a DenseNet model.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using pip:
   ```sh
   pip install tensorflow keras numpy matplotlib scikit-learn
   ```

## Usage

### 1. Data Loading and Preprocessing

Run `data_loading_and_preprocessing.py` to load and preprocess the CIFAR-10 dataset:

```sh
python data_loading_and_preprocessing.py
```

### 2. Model Building

Run `model_building.py` to define and compile the CNN architecture:

```sh
python model_building.py
```

### 3. Model Training

Run `model_training.py` to train the CNN model:

```sh
python model_training.py
```

### 4. Model Evaluation

Run `model_evaluation.py` to evaluate the model's performance and visualize predictions:

```sh
python model_evaluation.py
```

### 5. Utility Functions

`utility_functions.py` contains additional utility functions for creating models and visualizing the neural network. You can run and modify it as needed for your specific use cases.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CIFAR-10 dataset is publicly available and provided by the Canadian Institute for Advanced Research.
- This project uses TensorFlow and Keras for building and training the neural network.
