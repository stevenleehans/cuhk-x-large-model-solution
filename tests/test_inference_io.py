"""Input-contract tests for fresh paths, media, labels, and probability alignment."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from inference_io import read_questions,read_scores,prepare_media,safe_relative,decode_video

class InputTests(TestCase):
    def question(self,path='fresh/clip/Depth/Depth.mp4'):
        return dict(qa_id='new_id',source='HAU',category='single',path=path,
                    A='Walking',B='Running',C='',D='',answer='A',prediction='B')

    def test_fresh_paths_and_labels_removed(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'test_qa.csv';pd.DataFrame([self.question()]).to_csv(p,index=False)
            q=read_questions(p)
            self.assertEqual(q.clip_id.tolist(),['fresh/clip'])
            self.assertNotIn('answer',q);self.assertNotIn('prediction',q)
            pd.DataFrame([self.question(),self.question()]).to_csv(p,index=False)
            with self.assertRaises(ValueError):read_questions(p)

    def test_unsafe_paths_rejected(self):
        for path in ['/outside/file','../outside','fresh/../../secret','C:\\outside']:
            with self.assertRaises(ValueError):safe_relative(path)

    def test_scores_reordered_by_identity_and_bad_coverage_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'test_index.json').write_text(json.dumps(['b','a']))
            (root/'actions.json').write_text(json.dumps(['walk','jog']))
            np.save(root/'test_scores.npy',np.array([[.8,.2],[.1,.9]]))
            scores=read_scores(root,['a','b'],['walk','jog'])
            self.assertEqual(scores['a']['jog'],.9)
            with self.assertRaises(ValueError):read_scores(root,['a','c'],['walk','jog'])
            with self.assertRaises(ValueError):read_scores(root,['a','b'],['jog','walk'])
            np.save(root/'test_scores.npy',np.array([[float('nan'),.2],[.1,.9]]))
            with self.assertRaises(ValueError):read_scores(root,['a','b'],['walk','jog'])

    def test_valid_short_video_decodes_without_zero_fallback(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'fresh/clip/Depth';d.mkdir(parents=True)
            path=d/'Depth.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
            self.assertTrue(writer.isOpened())
            rng=np.random.default_rng(0)
            for _ in range(3):writer.write(rng.integers(1,255,(64,64,3),dtype=np.uint8))
            writer.release()
            arr,info=decode_video(path)
            self.assertEqual(arr.shape,(32,112,112));self.assertTrue(arr.any())
            self.assertEqual(info['decoded_frames'],3);self.assertEqual(info['duration_s'],0)
            p=root/'test_qa.csv';pd.DataFrame([self.question()]).to_csv(p,index=False)
            features=prepare_media(read_questions(p),root,root/'prepared')
            self.assertEqual(features.modality.tolist(),['Depth'])
            with self.assertRaises(FileExistsError):prepare_media(read_questions(p),root,root/'prepared')
