from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

