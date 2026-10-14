"""Regression checks for video evidence reaching the public prediction path."""
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from action_set import ActionSetModel
from fusion import FusedModel

class FusionTests(TestCase):
 def frame(self,source='HAU',category='multi'):
  return pd.DataFrame([dict(qa_id='q',clip_id='c',source=source,category=category,A='Walking',B='Running',C='',D='')])
 def test_public_predict_uses_canonical_video_scores(self):
  f=self.frame()
  with patch.object(ActionSetModel,'posteriors_batch',return_value={'c':{'Walking':.2,'Running':.8}}):
   self.assertEqual(FusedModel().predict(f),['B'])
   self.assertEqual(FusedModel({'c':{'walk':.99,'jog':.01}}).predict(f),['A'])
   self.assertEqual(FusedModel({'c':{'walk':.99,'jog':.01}},weight=0).predict(f),['B'])
 def test_harn_single_and_ambiguous_phrase(self):
  f=self.frame('HARn','single'); f.loc[0,'A']='walking'; f.loc[0,'B']='using a smartphone'
  scores={'walk':.1,'use_phone':.2,'games':.9}
  with patch.object(ActionSetModel,'posteriors_batch',return_value={}):
   self.assertEqual(FusedModel({'c':scores}).predict(f),['B'])
   self.assertEqual(FusedModel({'c':{'games':.9}}).predict(f),['A'])
 def test_invalid_scores_fail(self):
  for scores in [{'Walking':.9},{'walk':float('nan')},{'walk':1.1}]:
   with self.assertRaises(ValueError): FusedModel({'c':scores})
 def test_no_take_probability_preserves_emotion(self):
  f=self.frame(category='emotion')
  with patch.object(ActionSetModel,'predict',return_value=['B']):
   self.assertEqual(FusedModel(take_probs={'other':{'1':1}}).predict(f),['B'])
 def test_batch_and_single_apply_same_fusion(self):
  post={'Walking':.3,'Running':.6}; model=FusedModel({'c':{'walk':.9,'jog':.2}})
  with patch.object(ActionSetModel,'posteriors_batch',return_value={'c':post}), patch.object(ActionSetModel,'posteriors',return_value=post):
   self.assertEqual(model.posteriors(self.frame()),model.posteriors_batch(self.frame())['c'])

if __name__=='__main__':main()
