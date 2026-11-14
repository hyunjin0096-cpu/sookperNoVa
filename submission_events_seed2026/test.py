"""Inference only: no cache access, fitting, Test-fitted statistics or downloads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    import torch
    import joblib
    from image_encoder import load_encoder,extract
    from lastmile_runtime import build_features,EXTRA
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--image_root',required=True)
    parser.add_argument('--output',required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parent
    meta=json.loads((root/'weights/metadata.json').read_text())
    frame=pd.read_csv(a.input).reset_index(drop=True)
    needed=set(EXTRA+['sample_id','experiment_id','relative_time_s','image_path','meas_v','meas_a','meas_t1','meas_t2','meas_p'])
    if needed-set(frame): raise ValueError(f'Missing columns {needed-set(frame)}')
    if not len(frame) or frame[['sample_id','experiment_id','image_path']].isna().any().any(): raise ValueError('Empty/null input')
    if not frame.sample_id.is_unique or frame.duplicated(['experiment_id','relative_time_s']).any(): raise ValueError('Duplicate ID/time')
    if not np.isfinite(pd.to_numeric(frame.relative_time_s,errors='coerce')).all(): raise ValueError('Invalid times')
    for c in ['bat_class','bat_cc','soc_status','tr_method']:
        if (frame.groupby('experiment_id')[c].nunique(dropna=False)!=1).any(): raise ValueError('Changing conditions')
    if not frame.tr_method.isin([1,2]).all(): raise ValueError('Unknown method')
    original=frame.sample_id.copy()
    frame=frame.sort_values(['experiment_id','relative_time_s'],kind='stable').reset_index(drop=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder,device=load_encoder(str(root/'weights/resnet18.pth'))
    image_root=Path(a.image_root).resolve()
    if not image_root.is_dir(): raise FileNotFoundError(image_root)
    embeddings,stats=extract(frame,[image_root,image_root/'test'],encoder,device,batch_size=1)
    x=build_features(frame,embeddings,stats,meta['variant'])
    if list(x.columns)!=meta['features']: raise ValueError('Feature schema changed')
    probabilities=np.zeros((len(frame),6))
    methods=frame.tr_method.to_numpy(int)
    for fold,model_map in meta['models'].items():
        part=np.full_like(probabilities,np.nan)
        for method,filename in model_map.items():
            ids=np.arange(len(frame)) if method=='0' else np.flatnonzero(methods==int(method))
            if not len(ids): continue
            model=joblib.load(root/'weights'/filename);model.set_params(n_jobs=1)
            if list(model.feature_name_)!=meta['features'] or not np.array_equal(model.classes_,np.arange(6)):
                raise ValueError('Model schema mismatch')
            part[ids]=model.predict_proba(x.iloc[ids])
        if not np.isfinite(part).all() or (part<0).any(): raise ValueError('Invalid/unassigned predictions')
        probabilities+=part/len(meta['models'])
    if not np.allclose(probabilities.sum(1),1,atol=1e-5): raise ValueError('Probability sum mismatch')
    predicted=pd.Series(probabilities.argmax(1)+1,index=frame.sample_id)
    result=pd.DataFrame({'sample_id':original,'label':original.map(predicted).astype(int)})
    target=Path(a.output)
    if target.resolve()==Path(a.input).resolve(): raise ValueError('Output overwrites input')
    target.parent.mkdir(parents=True,exist_ok=True);result.to_csv(target,index=False)
    print(f'Saved {target}: {len(result)} rows')


if __name__=='__main__': main()
