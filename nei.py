import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Загрузка данных
df = pd.read_csv('/Users/nolvi/Downloads/TSLA.csv')

# Выделение признаков и целевой переменной
y = df['Close']
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1))  # Выходной слой без активации для регрессии

# Компиляция модели
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Первое предсказанное значение: {y_pred[0]}')
print(f'Первое значение тестовой выборки: {y_test.values[0]}')