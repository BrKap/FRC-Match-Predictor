from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

    def train(self, X, y):
        # SVM benefits a lot from scaling
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=True
            ))
        ])
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
