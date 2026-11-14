"""Original ThermalFusion features, frozen outer folds, inner-group stopping.

No test input. Does not overwrite original weights or cache. Run on competition
server. Fixed ImageNet cache is reused only with identical sample-ID order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

GAS_COLUMNS=['meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl',
             'meas_hf','meas_n2o','meas_no','meas_no2','meas_so2']

def attach_assignment(df, mapping):
    if list(mapping.columns)!=['experiment_id','fold']:
        raise ValueError('Expected exactly experiment_id,fold')
    if mapping.isna().any().any() or mapping.experiment_id.duplicated().any():
        raise ValueError('Null or duplicate experiment mapping')
    if not np.all(mapping.fold==mapping.fold.astype(int)) or set(mapping.fold)!=set(range(5)):
        raise ValueError('Expected integer fold IDs 0..4')
    if set(mapping.experiment_id)!=set(df.experiment_id):
        raise ValueError('Mapping must cover exactly all labeled experiments')
    if df.sample_id.isna().any() or not df.sample_id.is_unique:
        raise ValueError('Invalid sample IDs')
    if df.experiment_id.isna().any() or not df.label.isin(range(1,7)).all():
        raise ValueError('Invalid experiments/labels')
    if not np.isfinite(pd.to_numeric(df.relative_time_s,errors='coerce')).all():
        raise ValueError('Invalid timestamps')
    if df.duplicated(['experiment_id','relative_time_s']).any():
        raise ValueError('Duplicate experiment timestamp')
    for c in ['tr_method','bat_class','bat_cc','soc_status']:
        if (df.groupby('experiment_id')[c].nunique(dropna=False)!=1).any():
            raise ValueError(f'Inconsistent experiment condition {c}')
    result = df.copy()
    result['fold'] = result.experiment_id.map(mapping.set_index('experiment_id').fold).astype(int)
    for fold in range(5):
        for mask in [result.fold.eq(fold),result.fold.ne(fold)]:
            if set(result.loc[mask,'label'])!=set(range(1,7)):
                raise ValueError(f'Missing class in outer split {fold}')
    return result


def make_model(trees,threads,seed,num_leaves=31):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective='multiclass',num_class=6,n_estimators=trees,
        learning_rate=.03,max_depth=6,num_leaves=num_leaves,min_child_samples=40,
        colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_alpha=.1,reg_lambda=1.,
        random_state=seed,n_jobs=threads,verbosity=-1)


def probability(model,x):
    if not np.array_equal(model.classes_,np.arange(6)):
        raise ValueError('Unexpected probability class order')
    p = model.predict_proba(x)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid predictions')
    return p


def extra_for_variant(frame,original,variant):
    """Conditions ablation changes exactly two current, experiment-static inputs."""
    extra=list(original)
    if variant in ['conditions','conditions_gas','conditions_gas_history']:
        for column in ['bat_cc','soc_status']:
            if column not in frame:
                raise ValueError(f'Missing condition {column}')
            if column not in extra: extra.append(column)
        if variant in ['conditions_gas','conditions_gas_history']:
            for column in GAS_COLUMNS:
                if column not in frame: raise ValueError(f'Missing gas sensor {column}')
                if column not in extra: extra.append(column)
    elif variant!='original':
        raise ValueError(f'Unknown variant {variant}')
    return extra


def add_gas_status(x,frame):
    """Current observation only. Preserve zero as measured; never fill missing."""
    gas=frame[GAS_COLUMNS].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).reset_index(drop=True)
    result=x.copy()
    result['gas_zero_count']=gas.eq(0).sum(axis=1).to_numpy(dtype=np.float32)
    result['gas_missing_count']=gas.isna().sum(axis=1).to_numpy(dtype=np.float32)
    result['gas_all_zero']=(gas.notna().all(axis=1)&gas.eq(0).all(axis=1)).to_numpy(dtype=np.float32)
    return result


def add_history(x,frame):
    """22 causal fields. All assignments are positional and order invariant.

    Initial reference is the first supplied observation, not a full-experiment
    statistic. Past observations must be retained for prefix-equivalent input.
    """
    signals=['meas_v','meas_t1','meas_t2','meas_p','thermal_t']
    values=frame[signals].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).reset_index(drop=True)
    times=pd.to_numeric(frame.relative_time_s,errors='coerce').to_numpy(float)
    if not np.isfinite(times).all(): raise ValueError('Invalid history timestamps')
    names=[f'hist_{c}_{stat}' for c in signals for stat in ['from_start','past_max','drawdown','rate_per_s']]
    names+=['hist_voltage_initial_ratio','hist_gas_zero_duration_s']
    history=pd.DataFrame(np.nan,index=np.arange(len(frame)),columns=names,dtype=float)
    for indices in frame.groupby('experiment_id',sort=False).indices.values():
        ids=np.asarray(indices)
        ids=ids[np.argsort(times[ids],kind='stable')]
        t=times[ids]
        dt=np.r_[np.nan,np.diff(t)]
        if (dt[1:]<=0).any(): raise ValueError('Duplicate/nonincreasing history times')
        g=values.iloc[ids].reset_index(drop=True)
        for c in signals:
            v=g[c]
            maximum=v.cummax()
            history.loc[ids,f'hist_{c}_from_start']=(v-v.iloc[0]).to_numpy()
            history.loc[ids,f'hist_{c}_past_max']=maximum.to_numpy()
            history.loc[ids,f'hist_{c}_drawdown']=(maximum-v).to_numpy()
            history.loc[ids,f'hist_{c}_rate_per_s']=(v.diff()/dt).to_numpy()
        initial=g.meas_v.iloc[0]
        if np.isfinite(initial) and abs(initial)>1e-6:
            history.loc[ids,'hist_voltage_initial_ratio']=(g.meas_v/initial).to_numpy()
        gas=frame.iloc[ids][GAS_COLUMNS].apply(pd.to_numeric,errors='coerce')
        all_zero=gas.eq(0).all(axis=1).to_numpy()
        start=None
        durations=[]
        for i,zero in enumerate(all_zero):
            start=(t[i] if start is None else start) if zero else None
            durations.append(t[i]-start if zero else 0.)
        history.loc[ids,'hist_gas_zero_duration_s']=durations
    result=x.copy()
    for name in names: result[name]=history[name].to_numpy(dtype=np.float32)
    return result.replace([np.inf,-np.inf],np.nan)


def main():
    import joblib
    import sklearn
    import lightgbm
    from lightgbm import early_stopping,log_evaluation
    from sklearn.metrics import f1_score,classification_report,confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold
    from features import select_extra_columns,fuse
    from image_encoder import load_encoder,extract
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='/home/ubuntu/data/battery_data_light')
    parser.add_argument('--folds',default='folds_exp.csv')
    parser.add_argument('--out',default='frozen_cv_original')
    parser.add_argument('--cache-dir',default='cache')
    parser.add_argument('--weights',default=None,help='Custom encoder requires --rebuild-cache')
    parser.add_argument('--rebuild-cache',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--inner-seed',type=int,default=2026)
    parser.add_argument('--max-trees',type=int,default=1200)
    parser.add_argument('--patience',type=int,default=60)
    parser.add_argument('--num-leaves',type=int,default=31,
                        help='Only tree-complexity control; original baseline uses 31')
    parser.add_argument('--variant',choices=['original','conditions','conditions_gas','conditions_gas_history'],default='original',
                        help='conditions: +capacity/SOC; conditions_gas: additionally ten gases and three status fields')
    args=parser.parse_args()
    if min(args.threads,args.max_trees,args.patience)<1: parser.error('Positive counts required')
    if not 2<=args.num_leaves<=64: parser.error('--num-leaves must be 2..64 for max_depth=6')
    if args.weights and not args.rebuild_cache:
        parser.error('Custom encoder needs --rebuild-cache to avoid mixing encoder caches')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    data=Path(args.data)
    parts=[]
    for split in ['train','validation']:
        frame=pd.read_csv(data/split/(split+'.csv'))
        frame['source_split']=split
        parts.append(frame)
    if set(parts[0].experiment_id)&set(parts[1].experiment_id):
        raise ValueError('Train/Validation experiment overlap')
    df=attach_assignment(pd.concat(parts,ignore_index=True),pd.read_csv(args.folds))
    out.mkdir(parents=True,exist_ok=True)
    mapping=df[['experiment_id','fold']].drop_duplicates().sort_values('experiment_id')
    mapping.to_csv(out/'folds_used.csv',index=False)
    extra=extra_for_variant(parts[0],select_extra_columns(parts[0]),args.variant)
    print(f'Variant {args.variant}; extra columns:',extra,flush=True)
    cache_dir=Path(args.cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    encoder=device=None
    xs=[]
    for split,frame in zip(['train','validation'],parts):
        # Original cache remains untouched. Rebuilt caches go into this run.
        cache=cache_dir/f'images_{split}.joblib'
        if cache.exists() and not args.rebuild_cache:
            item=joblib.load(cache)
            if item['sample_ids']!=frame.sample_id.astype(str).tolist():
                raise ValueError(f'Cache ID/order mismatch: {cache}')
            embeddings,stats=item['embeddings'],item['stats']
            print(f'{split}: reuse {cache}',flush=True)
        else:
            if encoder is None: encoder,device=load_encoder(args.weights)
            embeddings,stats=extract(frame,[data/split,data],encoder,device)
            new_cache=out/'image_cache'
            new_cache.mkdir(exist_ok=True)
            joblib.dump({'sample_ids':frame.sample_id.astype(str).tolist(),
                'embeddings':embeddings,'stats':stats},new_cache/f'images_{split}.joblib')
        if len(embeddings)!=len(frame) or len(stats)!=len(frame): raise ValueError('Cache row mismatch')
        split_x=fuse(frame,embeddings,stats,extra,use_embeddings=True)
        if args.variant in ['conditions_gas','conditions_gas_history']: split_x=add_gas_status(split_x,frame)
        if args.variant=='conditions_gas_history': split_x=add_history(split_x,frame)
        xs.append(split_x)
    if list(xs[0].columns)!=list(xs[1].columns): raise ValueError('Feature columns differ by split')
    x=pd.concat(xs,ignore_index=True)
    if x.columns.duplicated().any(): raise ValueError('Duplicate feature names')
    if len(x)!=len(df): raise ValueError('Feature alignment error')
    y=df.label.to_numpy(int)-1
    folds=df.fold.to_numpy(int)
    oof=np.full((len(df),6),np.nan)
    logs=[]
    def metric(y,p): return float(f1_score(y,p.argmax(1),labels=list(range(6)),average='macro',zero_division=0))
    for fold in range(5):
        outer_train=np.flatnonzero(folds!=fold)
        outer_valid=np.flatnonzero(folds==fold)
        base=df.iloc[outer_train].reset_index(drop=True)
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=args.inner_seed)
        ia,ib=next(inner.split(np.zeros(len(base)),base.label,base.experiment_id))
        a,b=outer_train[ia],outer_train[ib]
        if set(df.iloc[a].experiment_id)&set(df.iloc[b].experiment_id): raise AssertionError('Inner overlap')
        if set(y[a])!=set(range(6)) or set(y[b])!=set(range(6)):
            raise ValueError(f'Inner split {fold} missing class; inspect split rather than silently change seed')
        roles=base[['experiment_id']].copy()
        roles['role']='inner_train'
        roles.loc[ib,'role']='inner_validation'
        roles.drop_duplicates().to_csv(out/f'inner_split_{fold}.csv',index=False)
        print(f'Fold {fold}: inner early stopping; train={len(a)}, valid={len(b)}',flush=True)
        selector=make_model(args.max_trees,args.threads,args.seed,args.num_leaves)
        selector.fit(x.iloc[a],y[a],eval_set=[(x.iloc[b],y[b])],eval_metric='multi_logloss',
            callbacks=[early_stopping(args.patience,verbose=False),log_evaluation(100)])
        trees=max(1,int(selector.best_iteration_))
        inner_f1=metric(y[b],probability(selector,x.iloc[b]))
        # Outer labels are never passed into this model's fitting/early stopping.
        model=make_model(trees,args.threads,args.seed,args.num_leaves)
        model.fit(x.iloc[outer_train],y[outer_train])
        p=probability(model,x.iloc[outer_valid])
        oof[outer_valid]=p
        train_f1=metric(y[outer_train],probability(model,x.iloc[outer_train]))
        valid_f1=metric(y[outer_valid],p)
        logs.append({'fold':fold,'trees':trees,'inner_f1':inner_f1,'train_f1':train_f1,
            'outer_f1':valid_f1,'train_rows':len(outer_train),'valid_rows':len(outer_valid)})
        joblib.dump(model,out/f'fold_{fold}.joblib')
        print(f'Fold {fold}: trees={trees}, train_F1={train_f1:.5f}, outer_F1={valid_f1:.5f}',flush=True)
    if not np.isfinite(oof).all(): raise AssertionError('Missing OOF predictions')
    results=df[['sample_id','experiment_id','source_split','relative_time_s','label','fold']].copy()
    results['prediction']=oof.argmax(1)+1
    for i in range(6): results[f'p_{i+1}']=oof[:,i]
    results.to_csv(out/'oof_predictions_SERVER_ONLY.csv',index=False)
    np.savez_compressed(out/'oof_probabilities_SERVER_ONLY.npz',probabilities=oof)
    pd.DataFrame(logs).to_csv(out/'fold_metrics.csv',index=False)
    exp_metrics=[]
    alias_map={}
    for source,g in df.groupby('source_split'):
        alias_map.update({eid:f'{source}_exp_{i:03d}' for i,eid in enumerate(sorted(g.experiment_id.unique()),1)})
    for eid,idx in df.groupby('experiment_id').indices.items():
        exp_metrics.append({'experiment':alias_map[eid],'fold':int(folds[idx[0]]),'rows':len(idx),
            'macro_f1_fixed6':metric(y[idx],oof[idx]),'errors':int((oof[idx].argmax(1)!=y[idx]).sum())})
    pd.DataFrame(exp_metrics).to_csv(out/'experiment_metrics.csv',index=False)
    report={'pooled_oof_macro_f1':metric(y,oof),'fold_mean':float(np.mean([r['outer_f1'] for r in logs])),
        'fold_std':float(np.std([r['outer_f1'] for r in logs])),'rows':len(df),'experiments':df.experiment_id.nunique(),
        'features':len(x.columns),'extra_columns':extra,'fold_results':logs,
        'confusion_matrix':confusion_matrix(y,oof.argmax(1),labels=list(range(6))).tolist(),
        'classification_report':classification_report(y+1,oof.argmax(1)+1,labels=list(range(1,7)),output_dict=True,zero_division=0),
        'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,
        'fold_file_sha256':hashlib.sha256(Path(args.folds).read_bytes()).hexdigest(),
        'variant':args.variant,'args':vars(args),
        'note':'CV only; feature additions determined by variant; no deployable final model or final refit produced.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'features.json').write_text(json.dumps(list(x.columns),indent=2),encoding='utf-8')
    print(f"Pooled OOF Macro F1: {report['pooled_oof_macro_f1']:.6f}",flush=True)
    print(f'Share {out}/summary.json and {out}/experiment_metrics.csv; keep SERVER_ONLY files on server.',flush=True)


if __name__=='__main__': main()
