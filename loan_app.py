import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

df = df.drop("no_of_dependents", axis=1)

df["loan_status"] = df["loan_status"].astype(str).str.lower().str.strip().map({
    "approved": 1,
    "rejected": 0
})

df["self_employed"] = df["self_employed"].astype(str).str.lower().str.strip().map({
    "no": 0,
    "yes": 1
})

df["education"] = df["education"].astype(str).str.lower().str.strip().map({
    "graduate": 1,
    "not graduate": 0
})

df = df.dropna()

X = df.drop(["loan_status", "loan_id"], axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))