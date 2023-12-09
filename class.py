import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Загрузка данных
df = pd.read_csv('/Users/nolvi/Downloads/TSLA.csv')

# Создание бинарного целевого признака: 1 - вырастет, 0 - упадет
df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Удаление последней строки, так как для нее нет следующего значения
df = df.dropna()

# Выделение признаков и целевого признака
y = df['Price_Up']
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))  # Добавление слоя Dropout
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Предсказание на тестовом наборе
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Преобразование вероятностей в бинарные предсказания

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Первое предсказанное значение (вероятность): {y_pred_prob[0][0]}')
print(f'Первое бинарное предсказание: {y_pred[0]}')
print(f'Первое фактическое значение тестовой выборки: {y_test.values[0]}')