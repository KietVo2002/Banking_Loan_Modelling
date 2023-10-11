# Import các thư viện cần thiết
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

bank = pd.read_csv("dulieunganhang.csv") 
bank.head()

X = bank.iloc[:, :-1].values
y = bank.iloc[:, -1].values


# Chia dữ liệu thành các tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tạo một bộ Gradient Boosting Regression với 100 cây quyết định và sử dụng hàm mất mát là mean squared error
gb = GradientBoostingRegressor(n_estimators=100, loss='squared_error', random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
gb.fit(X_train, y_train)

# Dự đoán giá trị đầu ra trên tập kiểm tra
y_pred = gb.predict(X_test)

# Đánh giá mô hình bằng cách tính toán mean squared error giữa giá trị đầu ra dự đoán và giá trị đầu ra thực tế trên tập kiểm tra
mse = mean_squared_error(y_test, y_pred)
# In ra giá trị mean squared error
print("Mean Squared Error: {:.2f}".format(mse))



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Create a logistic regression model
model = LogisticRegression()
model.fit (X_train, y_train)
y_pred = model.predict(X_test)


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model is: ',accuracy)