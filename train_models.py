import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import os

os.makedirs("models", exist_ok=True)

# بارگذاری داده (۸×۸)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# فصل ۱۴: Pipeline (نرمال‌سازی + مدل)
def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

# مدل‌های پایه (فصل ۱۱)
models = {
    "LogisticRegression": make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "KNN": make_pipeline(KNeighborsClassifier()),
    "SVM_RBF": make_pipeline(SVC(kernel="rbf", probability=True, random_state=42)),
    "DecisionTree": make_pipeline(DecisionTreeClassifier(random_state=42))
}

# فصل ۹: اعتبارسنجی k-تایی
print("=== Cross-Validation (5-fold) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# فصل ۱۶: تنظیم هایپرپارامتر
print("\n=== Hyperparameter Tuning ===")

# KNN
param_knn = {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}
grid_knn = GridSearchCV(models["KNN"], param_knn, cv=5, scoring="f1_macro", n_jobs=-1)
grid_knn.fit(X_train, y_train)
print("Best KNN:", grid_knn.best_params_, "F1:", grid_knn.best_score_)

# SVM
param_svm = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto", 0.01, 0.1]}
grid_svm = GridSearchCV(models["SVM_RBF"], param_svm, cv=5, scoring="f1_macro", n_jobs=-1)
grid_svm.fit(X_train, y_train)
print("Best SVM:", grid_svm.best_params_, "F1:", grid_svm.best_score_)

# Decision Tree
param_tree = {"clf__max_depth": [5, 10, 15, None], "clf__min_samples_split": [2, 5, 10]}
grid_tree = GridSearchCV(models["DecisionTree"], param_tree, cv=5, scoring="f1_macro", n_jobs=-1)
grid_tree.fit(X_train, y_train)
print("Best Tree:", grid_tree.best_params_, "F1:", grid_tree.best_score_)

# بهترین مدل‌ها بعد از تنظیم
best_models = {
    "LogisticRegression": models["LogisticRegression"],
    "KNN": grid_knn.best_estimator_,
    "SVM_RBF": grid_svm.best_estimator_,
    "DecisionTree": grid_tree.best_estimator_
}

# فصل ۱۵: Voting Classifier
voting = VotingClassifier(
    estimators=[
        ("lr", best_models["LogisticRegression"]),
        ("knn", best_models["KNN"]),
        ("svm", best_models["SVM_RBF"]),
        ("tree", best_models["DecisionTree"])
    ],
    voting="soft"
)
voting.fit(X_train, y_train)

# ارزیابی نهایی روی تست (فصل ۱۰)
print("\n=== Final Evaluation on Test Set ===")
y_pred = voting.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
print("\nConfusion Matrix 10×10:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ذخیره مدل‌ها
joblib.dump(voting, "models/voting_model.joblib")
joblib.dump(grid_svm.best_estimator_, "models/best_model.joblib")  # معمولاً SVM بهترین است
print("\nمدل‌ها ذخیره شدند.")