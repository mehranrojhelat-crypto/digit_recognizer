import numpy as np
import joblib
import os
import random
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.makedirs("models", exist_ok=True)

print("بارگذاری MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)  # نرمال‌سازی از اول

# داده بیشتر برای کیفیت بهتر
X, _, y, _ = train_test_split(X, y, train_size=20000, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print("ابعاد:", X_train.shape)


# ---------- انتخاب ویژگی ملایم‌تر ----------
class GeneticFeatureSelector:
    def __init__(self, X, y, pop_size=12, n_gen=8):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    def create_individual(self):
        # حداقل ۴۰۰ ویژگی نگه می‌داریم (ملایم‌تر)
        ind = np.ones(self.n_features, dtype=int)
        # فقط حدود ۳۰-۴۰٪ را تصادفی خاموش می‌کنیم
        off = np.random.choice(self.n_features, size=int(self.n_features * 0.35), replace=False)
        ind[off] = 0
        return ind

    def fitness(self, individual):
        selected = individual.astype(bool)
        if selected.sum() < 300:
            return 0.0
        X_sel = self.X[:, selected]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42))
        ])
        score = cross_val_score(model, X_sel, self.y, cv=self.cv, scoring="accuracy", n_jobs=-1).mean()
        return score

    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_ind, best_fit = None, -1

        for gen in range(self.n_gen):
            fitnesses = [self.fitness(ind) for ind in population]
            if max(fitnesses) > best_fit:
                best_fit = max(fitnesses)
                best_ind = population[np.argmax(fitnesses)].copy()
                print(f"نسل {gen+1}/{self.n_gen} | fitness: {best_fit:.4f} | ویژگی‌ها: {int(best_ind.sum())}")

            ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            next_pop = [ranked[i][0].copy() for i in range(2)]

            while len(next_pop) < self.pop_size:
                p1 = ranked[random.randint(0, 5)][0]
                p2 = ranked[random.randint(0, 5)][0]
                point = random.randint(1, self.n_features-1)
                child = np.concatenate([p1[:point], p2[point:]])
                # جهش کم
                mask = np.random.rand(self.n_features) < 0.02
                child[mask] = 1 - child[mask]
                if child.sum() < 300:
                    child[np.random.choice(self.n_features, 300, replace=False)] = 1
                next_pop.append(child)
            population = next_pop

        return best_ind


print("\n=== انتخاب ویژگی ملایم ===")
ga_fs = GeneticFeatureSelector(X_train, y_train)
best_mask = ga_fs.run()
print("ویژگی نهایی:", int(best_mask.sum()))

X_train_fs = X_train[:, best_mask.astype(bool)]
X_test_fs = X_test[:, best_mask.astype(bool)]
joblib.dump(best_mask, "models/feature_mask.joblib")


# ---------- هایپرپارامتر خیلی سریع ----------
print("\n=== تنظیم هایپرپارامتر سریع ===")
best_params = {
    "knn_n": 5,
    "knn_weights": "distance",
    "svm_C": 10,
    "svm_gamma": "scale",
    "tree_depth": 15,
    "tree_split": 2,
    "lr_C": 1.0
}

def build_models(params):
    return {
        "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=params["lr_C"], max_iter=300, random_state=42))]),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=params["knn_n"], weights=params["knn_weights"], n_jobs=-1))]),
        "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=params["svm_C"], gamma=params["svm_gamma"], probability=True, random_state=42))]),
        "tree": Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=params["tree_depth"], min_samples_split=params["tree_split"], random_state=42))])
    }

models = build_models(best_params)
voting = VotingClassifier(
    estimators=[("lr", models["lr"]), ("knn", models["knn"]), ("svm", models["svm"]), ("tree", models["tree"])],
    voting="soft", n_jobs=-1
)

print("در حال آموزش مدل نهایی...")
voting.fit(X_train_fs, y_train)

y_pred = voting.predict(X_test_fs)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1 Macro:", round(f1_score(y_test, y_pred, average="macro"), 4))
print(classification_report(y_test, y_pred))

joblib.dump(voting, "models/voting_model.joblib")
joblib.dump(best_params, "models/best_params.joblib")
print("\nمدل‌ها ذخیره شدند.")
