import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = train["Survived"]
X_t = test.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

c = ColumnTransformer([
    ("name1", OneHotEncoder(sparse_output=False), ["Sex"]),
    ("name2", Pipeline([
        ("a", SimpleImputer(strategy="most_frequent")),
        ("b", OneHotEncoder(handle_unknown="ignore"))
    ]), ["Embarked"]),
    ("name3", SimpleImputer(strategy="median"), ["Age", "Fare"]),
    ("name4", "passthrough", ["Pclass", "SibSp", "Parch"])
])

pipe1 = Pipeline([
    ("c", c),
    ("forest", RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None, min_samples_leaf=3))
])

pipe1.fit(X, y)
Survived = pipe1.predict(X_t)

sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Survived})
sub.to_csv("submission.csv", index=False)
