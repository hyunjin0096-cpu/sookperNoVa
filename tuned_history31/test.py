"""Standalone deployment entry point (packaged as test.py). No fitting/cache."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_frame(frame):
    from frozen_features import GAS_COLUMNS
    required=['sample_id','experiment_id','relative_time_s','image_path','meas_v','meas_a',
              'meas_t1','meas_t2','meas_p','thermal_t','indu_t','tr_method','bat_class',
              'bat_cc','soc_status',*GAS_COLUMNS]
    missing=set(required)-set(frame)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(frame)==0: raise ValueError('Empty input')
    if frame[['sample_id','experiment_id','image_path']].isna().any().any(): raise ValueError('Missing IDs/path')
    if not frame.sample_id.is_unique: raise ValueError('Duplicate sample IDs')
    if frame.duplicated(['experiment_id','relative_time_s']).any(): raise ValueError('Duplicate experiment/time')
    if not np.isfinite(pd.to_numeric(frame.relative_time_s,errors='coerce')).all(): raise ValueError('Invalid times')
    for col in ['bat_class','bat_cc','soc_status','tr_method']:
        if (frame.groupby('experiment_id')[col].nunique(dropna=False)!=1).any():
            raise ValueError(f'Changing experiment condition: {col}')


def build_input(frame,embeddings,stats,meta):
    from features import fuse
    from frozen_features import add_gas_status,add_history
    x=fuse(frame,embeddings,stats,meta['extra_columns'],use_embeddings=True)
    x=add_history(add_gas_status(x,frame),frame)
    if list(x.columns)!=meta['features']:
        raise ValueError('Feature name/order differs from training; refusing silent reindex')
    return x


def main():
    import joblib
    from image_encoder import load_encoder,extract
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--image_root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'weights/metadata.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(args.input).reset_index(drop=True)
    validate_frame(frame)
    # Sort before image extraction as well as causal processing; restore supplied order.
    original_ids=frame.sample_id.copy()
    frame=frame.sort_values(['experiment_id','relative_time_s'],kind='stable').reset_index(drop=True)
    image_root=Path(args.image_root).resolve()
    if not image_root.is_dir(): raise FileNotFoundError(image_root)
    encoder,device=load_encoder(str(root/'weights/resnet18.pth'))
    embeddings,stats=extract(frame,[image_root,image_root/'test'],encoder,device)
    x=build_input(frame,embeddings,stats,meta)
    p=np.zeros((len(frame),6),dtype=float)
    for filename in meta['models']:
        model=joblib.load(root/'weights'/filename)
        model.set_params(n_jobs=1)
        if not np.array_equal(model.classes_,np.arange(6)): raise ValueError('Class order mismatch')
        if list(model.feature_name_)!=meta['features']: raise ValueError('Model feature names mismatch')
        part=model.predict_proba(x)
        if part.shape!=p.shape or not np.isfinite(part).all() or (part<0).any():
            raise ValueError('Invalid probabilities')
        p+=part/len(meta['models'])
    if not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Unnormalized probabilities')
    lookup=pd.Series(p.argmax(1)+1,index=frame.sample_id)
    result=pd.DataFrame({'sample_id':original_ids,'label':original_ids.map(lookup).astype(int)})
    destination=Path(args.output)
    if destination.resolve()==Path(args.input).resolve(): raise ValueError('Output must not overwrite input')
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    print(f'Saved {destination}: {len(result)} rows; equal-weight mean of five fold probabilities')


if __name__=='__main__': main()
