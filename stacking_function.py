import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class StackingRegressor:
    def __init__(self, model1=LinearRegression(), model2=Lasso(), model3=MLPRegressor()):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def fit(self, Xtrain, ytrain):
        self.model1.fit(Xtrain, ytrain)
        self.model2.fit(Xtrain, ytrain)
        self.model3.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        ypred1 = self.model1.predict(Xtest)
        ypred2 = self.model2.predict(Xtest)
        ypred3 = self.model3.predict(Xtest)
        ypred = (ypred1 + ypred2 + ypred3) / 3
        return ypred

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Nhom 16_BTLCK/6_types_of_Stars.csv')

# Kiểm tra kiểu dữ liệu của từng cột và áp dụng LabelEncoder cho các cột có kiểu dữ liệu là chuỗi
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

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
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', solver='lbfgs', max_iter=2000, alpha=0.06, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# StackingRegressor
stacking_model = StackingRegressor(model1=linear_model, model2=lasso_model, model3=mlp_model)

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
