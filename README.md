# btc-trading

```
# Trading Prediction and Reinforcement Learning

This repository contains code for training a neural network model to predict cryptocurrency prices and using reinforcement learning to make trading decisions. It utilizes historical price data, performs data preprocessing, trains a neural network model, and uses the trained model for price prediction. It also includes a custom trading environment implemented using Gym, where a PPO agent is trained to make trading decisions based on the predicted prices.

## Getting Started

To get started with this code, follow the instructions below:

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- scikit-learn
- keras
- requests
- gym
- stable_baselines3

### Installation

1. Clone this repository to your local machine:

```
gh repo clone beyond-repair/btc-trading
```

2. Navigate to the project directory:

```
cd trading-prediction-rl
```

3. Install the required libraries:

```
pip install -r requirements.txt
```

### Usage

1. Prepare the Data:

   - Place your historical price data in a CSV file named `btc_prices.csv` in the project directory. The data should have a column named 'Close' containing the closing prices.

2. Neural Network Model:

   - Open the `neural_network.py` file and modify the parameters of the neural network model as needed.
   - Run the script to train the model and predict the next day's closing price:

   ```
   python neural_network.py
   ```

   The script will train the model, evaluate its performance, and print the predicted price.

3. API Integration:

   - Set your CoinAPI key by replacing `'YOUR_API_KEY'` with your actual API key in the `api_integration.py` file.
   - Run the script to retrieve additional price data from CoinAPI:

   ```
   python api_integration.py
   ```

   The script will send an API request, process the response data, and print information about each data point.

4. Reinforcement Learning:

   - Customize the trading logic in the `TradingEnvironment` class in the `rl_trading.py` file, implementing actions, state updates, and reward calculations based on current and next prices.
   - Run the script to train a PPO agent using the custom trading environment:

   ```
   python rl_trading.py
   ```

   The script will train the PPO agent and save the trained model.

5. Prediction and Trading:

   - Load the trained PPO model in the `rl_trading.py` file by specifying the model file path.
   - Run the script to make predictions and perform trading actions:

   ```
   python rl_trading.py
   ```

   The script will load the model, make predictions in the environment, and execute trading actions based on the predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code in this repository was inspired by various tutorials and examples available in the data science and reinforcement learning communities.
- Special thanks to the authors and contributors of the libraries used in this project.
```
