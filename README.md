# Instagram Fake Account Predictor

This project is a machine learning model designed to predict whether an Instagram account is real or fake. It uses a dataset of over 700 Instagram accounts and analyzes various features to determine the likelihood of an account being fake.

## Features and Technologies Used

- **Python Libraries**: 
  - `httpx`: For making HTTP requests to Instagram's API.
  - `jmespath`: For JSON parsing.
  - `pandas`, `numpy`: For data handling and manipulation.
  - `seaborn`, `matplotlib`: For data visualization.
  - `Scikit-Learn`: For building and training the machine learning model.
- **Machine Learning Algorithms**: 
  - Logistic Regression and Random Forest, combined into an ensemble model using a Voting Classifier.
- **Data Source**: Instagram account data, including follower counts, username details, and other account attributes.
- **CLI-based User Input**: Accepts a username input in the command line interface, fetches the account details from Instagram, and predicts if the account is fake or real.

## Getting Started

### Prerequisites

Make sure you have Python installed and install the following packages:

```pip install httpx jmespath kagglehub pandas numpy seaborn matplotlib scikit-learn```

## Running the Model

### Download the Dataset

The model uses a dataset from Kaggle. This script automatically downloads it via `kagglehub`. Ensure you have your Kaggle API key configured.

### Account Feature Extraction

The project includes a class, `igInfo`, that collects relevant features from a given Instagram account using `httpx`.  
The key features extracted are:
- Follower and following counts.
- Username and full name length, including whether they contain numbers.
- Account type (e.g., business account, private status).
- Account age (new vs. old).

### Training the Model

1. **Data Loading and Preprocessing**: The dataset is loaded, cleaned, and split into training and testing sets.
2. **Imbalanced Data Handling**: Class weights are calculated to address the imbalance between real and fake accounts in the dataset.
3. **Voting Classifier**: Two classifiers (Random Forest and Logistic Regression) are combined using a Voting Classifier to improve prediction accuracy.
4. **Hyperparameter Tuning**: The model’s parameters are optimized using Grid Search Cross-Validation to find the best combination of parameters for improved accuracy.

### Making Predictions

The program accepts an Instagram username through the CLI, fetches the account details, processes the data, and runs the trained model to predict if the account is fake or real.

## Usage

### Clone the Repository and Navigate to Project Directory

```git clone <repository-url>```
```cd <project-directory>```

### Run the predictor

Start the program and input an Instagram username for prediction.



