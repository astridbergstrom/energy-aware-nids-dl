from src.utils import set_seed
from codecarbon import EmissionsTracker
from src.data_preprocessing import preprocess_data
from src.dataset import create_dataloaders
from src.models import FFNN, LSTMModel, CNN1D
from src.train import train_model
from src.evaluate import evaluate_model

# Main thesis results are based on seed 42
set_seed(42)

# Data
train_X, train_y, test_X, test_y = preprocess_data(
    "data/KDDTrain+.txt",
    "data/KDDTest+.txt"
)

train_loader, test_loader = create_dataloaders(
    train_X, train_y, test_X, test_y
)

input_size = train_X.shape[1]

# ======================
# FFNN
# ======================
ffnn_model = FFNN(input_size)

train_tracker = EmissionsTracker(output_file="ffnn_train.csv", log_level="error")
train_tracker.start()

ffnn_model = train_model(ffnn_model, train_loader, epochs=25)

ffnn_train_emissions = train_tracker.stop()

test_tracker = EmissionsTracker(output_file="ffnn_test.csv", log_level="error")
test_tracker.start()

ffnn_acc, ffnn_prec, ffnn_rec, ffnn_f1 = evaluate_model(ffnn_model, test_loader)

ffnn_test_emissions = test_tracker.stop()

# ======================
# LSTM
# ======================
lstm_model = LSTMModel(input_size)

train_tracker = EmissionsTracker(output_file="lstm_train.csv", log_level="error")
train_tracker.start()

lstm_model = train_model(
    lstm_model,
    train_loader,
    epochs=25,
    is_lstm=True
)

lstm_train_emissions = train_tracker.stop()

test_tracker = EmissionsTracker(output_file="lstm_test.csv", log_level="error")
test_tracker.start()

lstm_acc, lstm_prec, lstm_rec, lstm_f1 = evaluate_model(
    lstm_model,
    test_loader,
    is_lstm=True
)

lstm_test_emissions = test_tracker.stop()

# ======================
# 1D-CNN
# ======================
input_size = train_X.shape[1]
cnn_model = CNN1D(input_size)

# Training
train_tracker = EmissionsTracker(output_file="cnn_train.csv", log_level="error")
train_tracker.start()

cnn_model = train_model(
    cnn_model,
    train_loader,
    epochs=25,
    is_cnn=True
)

cnn_train_emissions = train_tracker.stop()

# Test
test_tracker = EmissionsTracker(output_file="cnn_test.csv", log_level="error")
test_tracker.start()

cnn_acc, cnn_prec, cnn_rec, cnn_f1 = evaluate_model(
    cnn_model,
    test_loader,
    is_cnn=True
)

cnn_test_emissions = test_tracker.stop()

# ======================
# PRINT RESULTS
# ======================
print("\n--- FFNN ---")
print(f"Accuracy:  {ffnn_acc:.4f}")
print(f"Precision: {ffnn_prec:.4f}")
print(f"Recall:    {ffnn_rec:.4f}")
print(f"F1-score:  {ffnn_f1:.4f}")
print(f"Train CO2: {ffnn_train_emissions}")
print(f"Test CO2:  {ffnn_test_emissions}")

print("\n--- LSTM ---")
print(f"Accuracy:  {lstm_acc:.4f}")
print(f"Precision: {lstm_prec:.4f}")
print(f"Recall:    {lstm_rec:.4f}")
print(f"F1-score:  {lstm_f1:.4f}")
print(f"Train CO2: {lstm_train_emissions}")
print(f"Test CO2:  {lstm_test_emissions}")

print("\n--- 1D-CNN ---")
print(f"Accuracy:  {cnn_acc:.4f}")
print(f"Precision: {cnn_prec:.4f}")
print(f"Recall:    {cnn_rec:.4f}")
print(f"F1-score:  {cnn_f1:.4f}")
print(f"Train CO2: {cnn_train_emissions}")
print(f"Test CO2:  {cnn_test_emissions}")