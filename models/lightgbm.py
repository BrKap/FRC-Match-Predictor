import lightgbm as lgb


class LightGBMModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
