from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

