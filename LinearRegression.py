import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Hàm linear regression
class LinearRegression:
    def __init__(self, learning_rate=0.02, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.intercept = None
        self.coefficients = None

    def fit(self, X, y):
        # Thêm cột 1 vào ma trận X để tính toán hệ số điều chỉnh (intercept)
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Khởi tạo hệ số w ngẫu nhiên
        self.coefficients = np.random.randn(X.shape[1])

        # Gradient descent
        for _ in range(self.num_iterations):
            # Tính đầu ra dự đoán
            y_pred = np.dot(X, self.coefficients)

            # Tính sai số
            error = y_pred - y

            # Tính gradient
            gradient = np.dot(X.T, error) / X.shape[0]

            # Cập nhật hệ số w
            self.coefficients -= self.learning_rate * gradient

        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X, np.array([self.intercept] + list(self.coefficients)))

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

linear_model = LinearRegression(learning_rate=0.02, num_iterations=2000)
linear_model.fit(X_train_scaled, y_train)

y_pred_linear = linear_model.predict(X_test_scaled)

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

evaluate_model(y_test, y_pred_linear, 'Linear Regression')