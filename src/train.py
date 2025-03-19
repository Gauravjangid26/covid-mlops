import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "covid_toy.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f" The file '{file_path}' was not found. Check the path!")


# Load dataset
df = pd.read_csv(file_path)
print(df.head(3))
print(df.info())

#Feature engineering
print(df.isnull().sum())
df['fever'].fillna(df['fever'].mode()[0], inplace=True)
print(df.isnull().sum())

#data encoding
print(df.dtypes)
lb = LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['cough']=lb.fit_transform(df['cough'])
df['city']=lb.fit_transform(df['city'])
df['has_covid']=lb.fit_transform(df['has_covid'])

# Split data
X = df.drop("has_covid", axis=1)
y = df["has_covid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model=LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model in the correct directory
joblib.dump(model, os.path.join(BASE_DIR, "api/model.pkl"))
print("Model saved successfully!")