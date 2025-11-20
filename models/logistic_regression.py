from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                n_jobs=-1
            ))
        ])
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

