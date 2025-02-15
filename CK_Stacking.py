import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Nhom 16_BTLCK/6_types_of_Stars.csv')

# Kiểm tra kiểu dữ liệu của từng cột và áp dụng LabelEncoder cho các cột có kiểu dữ liệu là chuỗi
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

for column, le in label_encoders.items():
    print(f"Mapping for {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Chia dữ liệu thành tập train và tập test (70:30)
X = df.drop('Startype', axis=1)
y = df['Startype']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Lasso Regression
lasso_model = Lasso(alpha=0.05)
lasso_model.fit(X_train_scaled, y_train)

# Neural Network (MLPRegressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,50,), activation='tanh',solver='lbfgs', max_iter=2000, alpha=0.06, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# StackingRegressor
estimators = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.05)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,50,), activation='tanh',solver='lbfgs', max_iter=2000, alpha=0.06,random_state=42))
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10))
stacking_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập test
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_mlp = mlp_model.predict(X_test_scaled)
y_pred_stacking = stacking_model.predict(X_test_scaled)

# Đánh giá chất lượng mô hình
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # đặt squared=False để lấy căn bậc hai
    nse = 1 - (mse / np.var(y_true))
    
    print(f'{model_name} R2: {r2}')
    # print(f'{model_name} NSE: {nse}')
    # print(f'{model_name} MAE: {mae}')
    # print(f'{model_name} RMSE: {rmse}')

evaluate_model(y_test, y_pred_linear, 'Linear Regression')
evaluate_model(y_test, y_pred_lasso, 'Lasso Regression')
evaluate_model(y_test, y_pred_mlp, 'MLP Regressor')
evaluate_model(y_test, y_pred_stacking, 'Stacking Regressor')

# Tkinter GUI
class PredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Star Type Prediction")

        self.label_frame = ttk.LabelFrame(self.master, text="Enter New Data:")
        self.label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.r2_label = ttk.Label(self.master, text="R2 Score:")
        self.r2_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.mae_label = ttk.Label(self.master, text="MAE Score:")
        self.mae_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.rmse_label = ttk.Label(self.master, text="RMSE Score:")
        self.rmse_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")

        self.nse_label = ttk.Label(self.master, text="NSE Score:")
        self.nse_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")

        r2_stacking = r2_score(y_test, y_pred_stacking)
        self.r2_label["text"] = f"R2 Score Stacking: {r2_stacking:.4f}"

        mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
        self.mae_label["text"] = f"MAE Score Stacking: {mae_stacking:.4f}"

        rmse_stacking = mean_squared_error(y_test, y_pred_stacking, squared=False)
        self.rmse_label["text"] = f"RMSE Score Stacking: {rmse_stacking:.4f}"

        mse_stacking = mean_squared_error(y_test, y_pred_stacking)
        nse_stacking = 1 - (mse_stacking / np.var(y_test))
        self.nse_label["text"] = f"NSE Score Stacking: {nse_stacking:.4f}"

        self.inputs = {}
        self.entries = {}

        # Fill column names into the label_frame
        for i, column in enumerate(X.columns):
            ttk.Label(self.label_frame, text=column).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(self.label_frame)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.entries[column] = entry

        self.predict_button = ttk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.result_label = ttk.Label(self.master, text="")
        self.result_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.stacking_model = stacking_model

    def preprocess_input(self, input_data):
        input_data_df = pd.DataFrame([input_data])
        input_data_scaled = scaler.transform(input_data_df)
        return input_data_scaled

    def predict(self):
        new_data = {}
        for column, entry in self.entries.items():
            input_value = entry.get()

            # Kiểm tra giá trị NaN hoặc thiếu sót
            if input_value == "":
                self.result_label["text"] = "Vui lòng nhập giá trị cho tất cả các trường"
                return

            new_data[column] = float(input_value)

        new_data_scaled = self.preprocess_input(new_data)

        # Kiểm tra giá trị NaN trong dữ liệu đầu vào đã tiền xử lý
        if np.isnan(new_data_scaled).any():
            self.result_label["text"] = "Dữ liệu đầu vào không hợp lệ. Vui lòng kiểm tra giá trị của bạn."
            return

        # Thay đổi hình dạng dữ liệu đầu vào nếu cần
        if new_data_scaled.ndim == 1:
            new_data_scaled = new_data_scaled.reshape(1, -1)

        prediction = self.stacking_model.predict(new_data_scaled)[0]

        self.result_label["text"] = f"Dự đoán Loại Sao: {prediction:.2f}"

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
