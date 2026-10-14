"""Portable candidate pace decoder; structural-only action features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from action_set import ActionSetModel
from take_model import take_of


class ActionAwarePace:
    def _matrix(self, clips, features, posteriors):
        features = features.set_index('clip_id')
        counts = np.array([max(1, sum(posteriors[c].values())) for c in clips])
        duration = np.array([features.loc[c, 'duration_s'] for c in clips])
        scalar = np.log(np.maximum(duration / counts, 1e-6))[:, None]
        actions = np.array([[posteriors[c].get(a, 0) for a in self.vocabulary]
                            for c in clips]) / counts[:, None]
        return np.column_stack([scalar, actions])

    def fit(self, features, crossfit_posteriors):
        clips = sorted(c for c in crossfit_posteriors if take_of(c))
        self.vocabulary = sorted({a for c in clips for a in crossfit_posteriors[c]})
        self.model = LogisticRegression(C=1, max_iter=1000).fit(
            self._matrix(clips, features, crossfit_posteriors), [take_of(c) for c in clips])
        return self

    def predict_proba(self, features, posteriors):
        # Fresh anonymous IDs are accepted; only training uses path-derived takes.
        clips = sorted(set(posteriors) & set(features.clip_id))
        if not clips:
            return {}
        probabilities = self.model.predict_proba(self._matrix(clips, features, posteriors))
        return {c: dict(zip(self.model.classes_, p)) for c, p in zip(clips, probabilities)}


class PaceCandidate:
    """Wrap a fitted baseline without changing its non-emotion predictions."""
    def __init__(self, baseline, pace):
        self.baseline, self.pace = baseline, pace

    def predict(self, frame, feats):
        blind = frame.drop(columns=['answer'], errors='ignore')
        result = list(self.baseline.predict(blind, feats))
        # Bypass the installed video hook: training features are structural-only.
        post = ActionSetModel.posteriors_batch(self.baseline.structural, blind)
        probabilities = self.pace.predict_proba(feats, post)
        for i, row in enumerate(blind.to_dict('records')):
            if row['category'] == 'emotion':
                score = self.baseline.emotion.score(row, probabilities.get(row['clip_id']))
                result[i] = max(score, key=score.get)
        return result
