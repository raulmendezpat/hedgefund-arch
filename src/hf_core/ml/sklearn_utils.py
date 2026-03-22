from __future__ import annotations

class FunctionTransformerDense:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
