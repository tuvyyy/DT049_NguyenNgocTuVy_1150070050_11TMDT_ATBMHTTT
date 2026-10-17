from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dnn_model import FeedforwardDNNV2

# 1. Doc du lieu HAI test4 da xu ly
data = ROOT / "data/processed/hai_21_03/final_test"

X = np.load(data / "test4_features.npy", mmap_mode="r")
y = np.load(data / "test4_attack.npy", mmap_mode="r")
times = np.load(data / "test4_timestamps.npy", mmap_mode="r")

# 2. Nap mo hinh DNN da huan luyen
checkpoint = torch.load(
    ROOT / "models/v2_dnn/dnn_v2_seed42.pt",
    map_location="cpu",
    weights_only=True
)

model = FeedforwardDNNV2(
    hidden_activations=checkpoint["hidden_activations"],
    activation_status=checkpoint["activation_status"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

threshold = float(checkpoint["calibrated_threshold_test3"])

# 3. Lay su kien tan cong dau tien cua test4
start = int(np.flatnonzero(y[30:] == 1)[0] + 30)
end = start

while end + 1 < len(y) and y[end + 1] == 1:
    end += 1

attack_data = torch.from_numpy(
    np.array(X[start:end + 1], dtype=np.float32)
)

# 4. Cho AI du doan tung giay cua su kien
with torch.inference_mode():
    probabilities = model.predict_proba(
        attack_data
    ).flatten().numpy()

alarms = np.flatnonzero(probabilities > threshold)

# 5. In ket qua
print("\n===== DEMO PHAT HIEN TAN CONG HAI =====")
print("Dataset: HAI 21.03 / test4")
print("Model: DNN V2 - Seed 42")
print("So dac trung:", X.shape[1])
print("Nguong:", round(threshold, 6))

print("\n--- SU KIEN TAN CONG DAU TIEN ---")
print("Bat dau:", times[start])
print("Ket thuc:", times[end])
print("Thoi luong:", end - start + 1, "giay")

print("\nXac suat luc bat dau:", round(float(probabilities[0]), 6))
print("Xac suat cao nhat:", round(float(probabilities.max()), 6))

if len(alarms) > 0:
    delay = int(alarms[0])
    detected_row = start + delay

    print("\nKET QUA: PHAT HIEN TAN CONG")
    print("Thoi diem canh bao:", times[detected_row])
    print("Do tre phat hien:", delay, "giay")
    print("Xac suat tai canh bao:",
          round(float(probabilities[delay]), 6))
else:
    print("\nKET QUA: BO SOT TAN CONG")

print("\nDemo hoan tat. Khong thay doi model.")
