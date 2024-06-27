import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# サンプルCSVデータの作成
data = """
horse_speed,jockey_skill,track_condition,race_result
45,80,1,1
50,75,2,0
55,85,1,1
48,70,3,0
52,78,1,1
47,82,2,1
49,76,1,0
54,79,3,0
53,81,2,1
46,74,1,0
50,77,2,1
51,80,3,0
48,75,1,1
49,73,2,0
55,83,1,1
"""

# データをDataFrameに読み込む
import io
df = pd.read_csv(io.StringIO(data))

# データの前処理
X = df[['horse_speed', 'jockey_skill', 'track_condition']]
y = df['race_result']

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築
model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルのトレーニング
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# モデルの評価
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# 新しいレースデータで予測
new_data = pd.DataFrame({
    'horse_speed': [45, 50, 55],
    'jockey_skill': [80, 75, 85],
    'track_condition': [1, 2, 1]
})

predictions = model.predict(new_data)

# 新しいレースデータと予測結果をCSVファイルに保存
new_data['predictions'] = predictions
new_data.to_csv('predictions.csv', index=False)

print('Predictions saved to predictions.csv:')
print(new_data)
