import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import requests
import gym
from stable_baselines3 import PPO


# Load the historical price data from a CSV file
data = pd.read_csv('btc_prices.csv')

# Prepare the data for training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Split the data into training and testing sets
X = []
y = []
window_size = 30  # Number of previous days' prices to use as input
for i in range(window_size, len(data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=window_size))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model's performance
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

# Predict the next day's closing price
last_day_prices = data['Close'].tail(window_size).values
last_day_scaled = scaler.transform(last_day_prices.reshape(-1, 1))
next_day_scaled = model.predict(last_day_scaled.reshape(1, -1))
next_day_price = scaler.inverse_transform(next_day_scaled)
print("Predicted BTC Price for the Next Day:", next_day_price)

# Set your API key
api_key = ' 5F1A4F95-76DE-4B97-9EB2-7CACBCA7555A'

# Define the API endpoint and parameters
endpoint = 'https://rest.coinapi.io/v1/ohlcv/BTC/USD/history'
params = {
    'period_id': '1DAY',  # Time period (e.g., 1DAY, 1HRS, 5MINS)
    'limit': 10  # Number of data points to retrieve
}

# Set the request headers with your API key
headers = {
    'X-CoinAPI-Key': api_key
}

# Send the API request
response = requests.get(endpoint, params=params, headers=headers)

# Check the response status code
if response.status_code == 200:
    # Process the response JSON data
    data = response.json()
    for item in data:
        timestamp = item['time_period_start']
        open_price = item['price_open']
        high_price = item['price_high']
        low_price = item['price_low']
        close_price = item['price_close']
        print(f"Timestamp: {timestamp}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# Create a custom trading environment
class TradingEnvironment(gym.Env):
    def __init__(self):
        # Initialize your trading environment
        # Set up variables, data, and any necessary components
        
    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        
    def step(self, action):
        # Take an action in the environment
        # Perform necessary calculations, update state, and return the next observation, reward, and done flag

# Create an instance of your custom trading environment
env = TradingEnvironment()

# Define the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the PPO agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_trained_model")

# Load the trained model
loaded_model = PPO.load("ppo_trained_model")

# Use the loaded model to make predictions
obs = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs)
    obs, reward, done, _ = env.step(action)
    # Perform necessary actions based on the prediction (e.g., execute a trade)

# Close the environment
env.close()
import gym
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.action_space = spaces.Discrete(3)  # Assuming 3 discrete actions: 0 (Sell), 1 (Hold), 2 (Buy)
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size,), dtype=np.float32)
        self.current_step = window_size

    def reset(self):
        self.current_step = window_size
        return self._get_observation()

    def step(self, action):
        # Perform necessary calculations, update state, and return the next observation, reward, and done flag
        if self.current_step >= len(self.data):
            # If we have reached the end of the data, terminate the episode
            return self._get_observation(), 0, True, {}

        # Get the current price and the next price
        current_price = self.data[self.current_step - 1]
        next_price = self.data[self.current_step]

        # Take action and calculate the reward
        reward = self._take_action(action, current_price, next_price)

        # Update the current step
        self.current_step += 1

        # Return the next observation, reward, and done flag
        return self._get_observation(), reward, False, {}

    def _get_observation(self):
        # Get the observation for the current step
        observation = self.data[self.current_step - window_size : self.current_step]
        return observation

    def _take_action(self, action, current_price, next_price):
        # Implement your logic to take an action based on the current and next prices
        # Update your portfolio, execute trades, calculate rewards, etc.
        # Return the reward for the action taken
        reward = 0
        # Your implementation here...
        return reward


