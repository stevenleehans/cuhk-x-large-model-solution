"""Public prediction must use the current split's video scores."""
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from action_set import ActionSetModel
from pipeline import Pipeline

class PipelineVideoTests(TestCase):
    def test_replacing_training_scores_changes_test_prediction(self):
        frame = pd.DataFrame([dict(qa_id='q', clip_id='test_clip', source='HAU',
                                   category='multi', A='Walking', B='Running', C='', D='')])
        def posterior(_self, _frame):
            return {'test_clip': {'Walking': .2, 'Running': .8}}
        with patch.object(ActionSetModel, 'posteriors_batch', posterior):
            model = Pipeline(video_scores={'train_clip': {'walk': .99, 'jog': .01}})
            self.assertEqual(model.predict(frame), ['B'])
            model.video_scores = {'test_clip': {'walk': .99, 'jog': .01}}
            self.assertEqual(model.predict(frame), ['A'])

    def test_pickle_roundtrip_preserves_dynamic_score_switch(self):
        import pickle
        frame = pd.DataFrame([dict(qa_id='q', clip_id='test_clip', source='HAU',
                                   category='multi', A='Walking', B='Running', C='', D='')])
        def posterior(_self, _frame):
            return {'test_clip': {'Walking': .2, 'Running': .8}}
        model = Pipeline(video_scores={'train_clip': {'walk': .99, 'jog': .01}})
        restored = pickle.loads(pickle.dumps(model))
        with patch.object(ActionSetModel, 'posteriors_batch', posterior):
            self.assertEqual(restored.predict(frame), ['B'])
            restored.video_scores = {'test_clip': {'walk': .99, 'jog': .01}}
            self.assertEqual(restored.predict(frame), ['A'])
