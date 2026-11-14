"""Method-B cadence diagnostics for saved V3-B physics models (no retraining).

For each cadence: remove rows FIRST, rebuild all time-dependent base and physics
features SECOND, infer THIRD. Comparisons on same target rows are diagnostic.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import model as B
import physics_features as F

LABELS=np.arange(1,7)

def metrics(y,p):
    return {'n':len(y),'accuracy':accuracy_score(y,p),
            'macro_precision':precision_score(y,p,labels=LABELS,average='macro',zero_division=0),
            'macro_recall':recall_score(y,p,labels=LABELS,average='macro',zero_division=0),
            'macro_f1':f1_score(y,p,labels=LABELS,average='macro',zero_division=0)}

def subsample_fixed(df,seconds):
    # Same rule as prior standard_compare.py; not origin-from-experiment-start.
    t=pd.to_numeric(df.relative_time_s,errors='raise').to_numpy()
    return df.loc[np.isclose(np.mod(t,seconds),0.0,atol=1e-8)].reset_index(drop=True)

def subsample_mixed(df,seed,gaps=(5,10,15)):
    ts=pd.to_numeric(df.relative_time_s,errors='raise').to_numpy(dtype=float)
    mask=np.zeros(len(df),bool)
    for exp,idx in df.groupby('experiment_id',sort=False).indices.items():
        seed_ex=int.from_bytes(hashlib.sha256(f'{seed}|{exp}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed_ex)
        idx=np.asarray(idx,dtype=int)
        idx=idx[np.argsort(ts[idx],kind='mergesort')]
        t=ts[idx];i=0
        mask[idx[0]]=True
        while i<len(idx)-1:
            target=t[i]+int(rng.choice(gaps))
            j=int(np.searchsorted(t,target,side='left'))
            if j>=len(idx):break
            mask[idx[j]]=True;i=j
    return df.loc[mask].reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--image-cache',type=Path,default=Path('cache/v3b_cnn'))
    ap.add_argument('--spatial-cache',type=Path,default=Path('cache/v3b_physics'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/v3b_physics/cadence_metrics.csv'))
    ap.add_argument('--mixed-seeds',nargs='+',type=int,default=[2026,2027,2028])
    args=ap.parse_args()
    bundle=joblib.load(args.model)
    if bundle.get('origin')!='teammate_v3B_plus_causal_physics_v1':
        raise ValueError('Not a physics bundle')
    df=pd.read_csv(args.data_dir/'validation'/'validation.csv',
                    dtype={'sample_id':'string','experiment_id':'string'}).reset_index(drop=True)
    cache=args.image_cache/'img_features_v2_validation.csv'
    if cache.exists():
        img=pd.read_csv(cache,dtype={'sample_id':'string'},float_precision='round_trip').set_index('sample_id')
        if list(img.columns)!=B.IMAGE_STATS or len(img)!=len(df) or set(img.index)!=set(df.sample_id):
            raise ValueError('Invalid B image stats cache; delete it and recompute with train_physics.py')
    else:
        img=B.extract_image_features(df,B.resolve_image_paths(df,[args.data_dir/'validation',args.data_dir]),n_jobs=2)
    spatial=None
    if 'image_spatial' in bundle['groups']:
        p=args.spatial_cache/f'spatial_{F.SPATIAL_VERSION}_validation.csv'
        spatial=F.extract_spatial(df,B.resolve_image_paths(df,[args.data_dir/'validation',args.data_dir]),p)
    def predict(sub):
        sub=sub.reset_index(drop=True)
        im=img.reindex(sub['sample_id'].astype(str).to_numpy())
        x=B.build_features(sub,im,bundle['spec'])
        if bundle['groups']:
            ph=F.physics_features(sub,bundle['groups'],spatial=spatial,
                                  thresholds=bundle.get('train_thresholds'))
            x=pd.concat([x.reset_index(drop=True),ph.reset_index(drop=True)],axis=1)
        prob=bundle['model'].predict_proba(x[bundle['features']])
        return np.asarray(bundle['model'].classes_)[prob.argmax(axis=1)]
    true=pd.to_numeric(df.label).to_numpy(dtype=int)
    full=predict(df)
    full_map=pd.Series(full,index=df.sample_id.astype(str).to_numpy())
    rows=[]
    def eval_one(name,sub):
        y=pd.to_numeric(sub.label).to_numpy(dtype=int)
        p=predict(sub)
        m=metrics(y,p); rows.append({'subset':name,**m})
        print(f'{name:22} | N={m["n"]:5} | Accuracy={m["accuracy"]:.4f} | Macro F1={m["macro_f1"]:.4f}')
        fullrows=full_map.loc[sub.sample_id.astype(str).to_numpy()].to_numpy(dtype=int)
        ref=metrics(y,fullrows)
        print(f'  same rows full history | Macro F1={ref["macro_f1"]:.4f} | prediction flips={int(np.sum(fullrows!=p))}')
        rows.append({'subset':name+'_full_history_reference',**ref})
        return p
    eval_one('original_5s',df)
    for sec in (10,15):eval_one(f'{sec}s_method_B',subsample_fixed(df,sec))
    for seed in args.mixed_seeds:eval_one(f'mixed_seed_{seed}',subsample_mixed(df,seed))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print('[SAVED]',args.output)

if __name__=='__main__':main()
