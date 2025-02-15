import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Nhom 16_BTLCK/6_types_of_Stars.csv')

# Kiểm tra kiểu dữ liệu của từng cột và áp dụng LabelEncoder cho các cột có kiểu dữ liệu là chuỗi
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])
        
# Chia dữ liệu thành tập train và tập test (70:30)
X = df.drop('Startype', axis=1)
y = df['Startype']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network (MLPRegressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,50,), activation='tanh',solver='lbfgs', max_iter=2000, alpha=0.06, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập test
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Đánh giá chất lượng mô hình
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # đặt squared=False để lấy căn bậc hai
    nse = 1 - (mse / np.var(y_true))
    
    print(f'{model_name} R2: {r2}')
    print(f'{model_name} NSE: {nse}')
    print(f'{model_name} MAE: {mae}')
    print(f'{model_name} RMSE: {rmse}')

evaluate_model(y_test, y_pred_mlp, 'MLP Regressor')