from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        # Tree-based model, scaling not required
        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # XGBoost returns both class probabilities; we want P(red_win=1)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict_with_threshold(self, X, threshold):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
