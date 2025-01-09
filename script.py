# Necessary imports
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Set PyTorch manual seed
torch.manual_seed(42)

# Hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 50
RNN_LAYERS = 2
HIDDEN_SIZE = 64
TRAIN_TEST_SPLIT = 0.90
LOOK_BACK = 7
BATCH_SIZE = 64
PLOT_STEPS = 1
PRINT_STEPS = 5


df = pd.read_csv('data/NVDA.csv')

dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in list(df['Date'])]
prices = [float(p) for p in list(df['Close'])]

plt.title("NVDA Stock Data")
plt.xlabel("Year")
plt.ylabel("Price")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()
plt.plot(dates, prices);

def prepare_lookback_df(df):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, LOOK_BACK + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

shifted_df = prepare_lookback_df(df[['Date', 'Close']])

shifted_np = shifted_df.to_numpy()
scaler = MinMaxScaler((-1, 1))
shifted_np = scaler.fit_transform(shifted_np)

# Since LSTMs need to proces older data first and gradually time step into newer data
X = torch.from_numpy(dc(np.flip(shifted_np[:, 1:], axis=1))).type(torch.float32)
y = torch.from_numpy(shifted_np[:, 0]).type(torch.float32)

cutoff = int(len(X) * TRAIN_TEST_SPLIT)

X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]

# Need to explicitly mention an input dimension
# In this case, the sequence length (L) is 7 and input shape (H_in) is 1
X_train = X_train.unsqueeze(dim=2)
X_test = X_test.unsqueeze(dim=2)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class StockLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.classifier(out)

        return out


model = StockLSTM(
    input_size=1,
    output_size=1,
    num_layers=RNN_LAYERS,
    hidden_size=HIDDEN_SIZE
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)


def train(model, dataloader, loss_fn, optimizer):
    cur_loss = 0.0
    model.train()

    for X, y in dataloader:
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        cur_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cur_loss /= len(dataloader)
    
    return cur_loss

def test(model, dataloader, loss_fn):
    cur_loss = 0.0
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            cur_loss += loss.item()

        cur_loss /= len(dataloader)

    return cur_loss


all_train_losses, all_test_losses = [], []
running_train_loss, running_test_loss = 0.0, 0.0

for epoch in range(EPOCHS):
    cur_train_loss = train(model, train_dataloader, loss_fn, optimizer)
    cur_test_loss = test(model, test_dataloader, loss_fn, optimizer)

    running_train_loss += cur_train_loss
    running_test_loss += cur_test_loss

    if epoch % PLOT_STEPS == 0:
        all_train_losses.append(running_train_loss / PLOT_STEPS)
        all_test_losses.append(running_test_loss / PLOT_STEPS)

        running_train_loss, running_test_loss = 0.0, 0.0

    if epoch % PRINT_STEPS == 0:
        print(f"Epoch: {epoch} | Train Loss: {cur_train_loss:.4f} | Test Loss: {cur_test_loss:.4f}")


# Plot train loss curve
plt.title("Train Loss Curve")
plt.plot(all_train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss');

# Plot test loss curve
plt.title("Test Loss Curve")
plt.plot(all_test_losses);
plt.xlabel('Epoch')
plt.ylabel('Loss');

model.eval()

with torch.inference_mode():
    pred = model(X_train)

plt.title("Training Data")
plt.plot(y_train, label='Actual Price')
plt.plot(pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Scaled Close')
plt.legend()
plt.show();

with torch.inference_mode():
    pred = model(X_test)

plt.title("Test Data")
plt.plot(y_test, label='Actual Price')
plt.plot(pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Scaled Close')
plt.legend()
plt.show();

with torch.inference_mode():
    pred = model(X.unsqueeze(dim=2))

plt.title("NVDA Stock vs. Predictions")
plt.plot(y, label='Actual Price')
plt.plot(pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Scaled Close')
plt.legend()
plt.show();

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "nvidia-lstm-model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)


# Create a new instance of loaded model
loaded_model = StockLSTM(
    input_size=1,
    output_size=1,
    num_layers=RNN_LAYERS,
    hidden_size=HIDDEN_SIZE
)

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=False))

# Evaluate loaded model to ensure that results are similar to base model
base_model_loss = test(model, test_dataloader, loss_fn, optimizer)
loaded_model_loss = test(loaded_model, test_dataloader, loss_fn, optimizer)

# Check if model results are similar
is_close = torch.isclose(
    torch.tensor(base_model_loss, dtype=torch.float32),
    torch.tensor(loaded_model_loss, dtype=torch.float32),
    atol=1e-04 # Absolute error tolerance
).item()

assert is_close
