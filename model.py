import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop useless columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode categorical columns
encoders = {}
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Features & target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Features:")
print(feat_imp.sort_values(ascending=False).head(10))

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(list(X.columns), open("features.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
print("\nModel saved!")