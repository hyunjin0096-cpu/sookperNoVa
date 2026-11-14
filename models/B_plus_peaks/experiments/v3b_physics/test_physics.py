"""Official-style test interface: --input CSV --image_root DIR --output CSV.

Causal features are generated exclusively from supplied rows within each experiment;
subsetting future rows or changing input ordering must not change older predictions.
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import model as B
import physics_features as F


def infer(frame, image_root, bundle, blank=False):
    if 'sample_id' not in frame or frame['sample_id'].duplicated().any():
        raise ValueError('Input sample_id must be present and unique')
    if blank:
        with tempfile.TemporaryDirectory() as temp:
            im=Path(temp)/'blank.png';Image.new('RGB',(128,128),(0,0,0)).save(im)
            vals=B._image_stats(im)
            stats=pd.DataFrame([vals]*len(frame),columns=B.IMAGE_STATS,
              index=pd.Index(frame.sample_id.astype(str).to_numpy(),name='sample_id'))
            spatial=None
            if 'image_spatial' in bundle['groups']:
                sv=F.spatial_stats(im)
                spatial=pd.DataFrame([sv]*len(frame),columns=F.SPATIAL_NAMES,
                      index=pd.Index(frame.sample_id.astype(str).to_numpy(),name='sample_id'))
    else:
        paths=B.resolve_image_paths(frame,[image_root,image_root.parent])
        stats=B.extract_image_features(frame,paths,n_jobs=2)
        spatial=F.extract_spatial(frame,paths,n_jobs=2) if 'image_spatial' in bundle['groups'] else None
    Xbase=B.build_features(frame,stats,bundle['spec'])
    phys=F.physics_features(frame,bundle['groups'],spatial=spatial,
                             thresholds=bundle.get('train_thresholds')) if bundle['groups'] else pd.DataFrame(index=frame.index)
    X=pd.concat([Xbase.reset_index(drop=True),phys.reset_index(drop=True)],axis=1)
    missing=set(bundle['features'])-set(X.columns)
    if missing:raise ValueError(f'Model features missing: {sorted(missing)}')
    proba=bundle['model'].predict_proba(X[bundle['features']])
    pred=np.asarray(bundle['model'].classes_)[proba.argmax(axis=1)]
    return pd.DataFrame({'sample_id':frame['sample_id'].astype('string').to_numpy(),
                         'label':pred.astype(int)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image_root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--sample_submission',type=Path,default=None)
    p.add_argument('--blank_image',action='store_true')
    args=p.parse_args()
    bundle=joblib.load(args.model)
    if bundle.get('origin')!='teammate_v3B_plus_causal_physics_v1':
        raise ValueError('This entrypoint expects v3B physics model bundle')
    frame=pd.read_csv(args.input,dtype={'sample_id':'string','experiment_id':'string'})
    pred=infer(frame,args.image_root,bundle,blank=args.blank_image)
    if args.sample_submission:
        sample=pd.read_csv(args.sample_submission,dtype={'sample_id':'string'})
        assert len(sample)==len(pred) and set(sample.sample_id)==set(pred.sample_id), 'Submission IDs mismatch'
        # Input order is the reference unless competition explicitly specifies otherwise.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pred.to_csv(args.output,index=False)
    print('[SAVED]',args.output,'rows',len(pred),'blank_image',args.blank_image)

if __name__=='__main__':main()
