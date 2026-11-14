"""Physics-informed, leak-aware ablations of teammate V3-B (LightGBM, 190 features).

No private data leaves this process. GroupKFold grouped by ORIGINAL experiment ID.
Model/hyperparameters/augmentation held fixed. Each experiment is independently
fit; feature choice must be based on TRAIN OOF CV, not repeated validation tuning.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import model as B
import physics_features as F

LABELS=np.arange(1,7)

def score(y,p):return float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0))

def image_stats(data, split, frame, cache_dir):
    p=cache_dir/f'img_features_v2_{split}.csv'
    if p.exists():
        im=pd.read_csv(p,dtype={'sample_id':'string'},float_precision='round_trip').set_index('sample_id')
        if list(im.columns)==B.IMAGE_STATS and len(im)==len(frame) and set(im.index)==set(frame.sample_id):
            print('[IMG] original B cache reused:',p)
            return im
    paths=B.resolve_image_paths(frame,[data/split,data])
    im=B.extract_image_features(frame,paths,n_jobs=2)
    p.parent.mkdir(parents=True,exist_ok=True)
    im.to_csv(p)
    return im

def build_parts(data, image_cache, spatial_cache, groups_union):
    train=pd.read_csv(data/'train'/'train.csv',dtype={'sample_id':'string','experiment_id':'string'})
    val=pd.read_csv(data/'validation'/'validation.csv',dtype={'sample_id':'string','experiment_id':'string'})
    assert set(train.experiment_id).isdisjoint(val.experiment_id), 'Train/validation experiment overlap'
    spec=B.infer_feature_spec(train, interactions=False)
    imtr=image_stats(data,'train',train,image_cache)
    imval=image_stats(data,'validation',val,image_cache)
    spatialtr=spatialval=None
    if 'image_spatial' in groups_union:
        spatialtr=F.extract_spatial(train,B.resolve_image_paths(train,[data/'train',data]),
                 cache_path=spatial_cache/f'spatial_{F.SPATIAL_VERSION}_train.csv')
        spatialval=F.extract_spatial(val,B.resolve_image_paths(val,[data/'validation',data]),
                 cache_path=spatial_cache/f'spatial_{F.SPATIAL_VERSION}_validation.csv')
    frames=[train,B.subsample_experiments(train,2),B.subsample_experiments(train,3)]
    Xbase=pd.concat([B.build_features(f,imtr,spec) for f in frames],ignore_index=True)
    Xva=B.build_features(val,imval,spec)
    if len(Xbase.columns)!=190:
        print(f'[WARNING] B feature count is {len(Xbase.columns)}; expected 190. Verify training data/spec!')
    Y=pd.concat([pd.to_numeric(f['label']).astype(int) for f in frames],ignore_index=True)
    groups=pd.concat([f.experiment_id.str.replace(r'__every\d+$','',regex=True)
                     for f in frames],ignore_index=True)
    orig=np.zeros(len(Xbase),dtype=bool);orig[:len(train)]=True
    feats_train={};feats_val={}
    regular=[g for g in groups_union if g!='train_quantiles']
    if regular:
        # Compute each group independently; addition order does not affect its feature values.
        for g in regular:
            parts=[F.physics_features(f,[g],spatial=spatialtr) for f in frames]
            feats_train[g]=pd.concat(parts,ignore_index=True)
            feats_val[g]=F.physics_features(val,[g],spatial=spatialval).reset_index(drop=True)
            print(f'[FEATURE] {g}: {len(feats_train[g].columns)} new columns')
    return dict(train=train,val=val,frames=frames,imtr=imtr,imval=imval,
                spec=spec,Xbase=Xbase,Xva=Xva,y=Y,exp_groups=groups,orig=orig,
                feats_train=feats_train,feats_val=feats_val,
                spatialtr=spatialtr,spatialval=spatialval)

def form_X(b, group_list, threshold_spec=None, is_validation=False):
    base=b['Xva'] if is_validation else b['Xbase']
    blocks=[base]
    maps=b['feats_val'] if is_validation else b['feats_train']
    for g in F.GROUPS:
        if g not in group_list or g=='train_quantiles':continue
        blocks.append(maps[g])
    if 'train_quantiles' in group_list:
        if threshold_spec is None:raise ValueError('Missing fold-specific quantiles')
        if is_validation:
            tf=F.physics_features(b['val'],['train_quantiles'],thresholds=threshold_spec)
        else:
            tf=pd.concat([F.physics_features(f,['train_quantiles'],thresholds=threshold_spec)
                          for f in b['frames']],ignore_index=True)
        blocks.append(tf.reset_index(drop=True))
    out=pd.concat(blocks,axis=1)
    if out.columns.duplicated().any():raise AssertionError('duplicate columns')
    return out

def evaluate_details(y,p,frame,model_name,out):
    print('\n======',model_name,'validation ======')
    print('Accuracy',f'{accuracy_score(y,p):.4f}','Macro F1',f'{score(y,p):.4f}')
    print(classification_report(y,p,labels=LABELS,digits=4,zero_division=0))
    cm=confusion_matrix(y,p,labels=LABELS)
    print(pd.DataFrame(cm,index=[f'True_{j}' for j in LABELS],columns=[f'Pred_{j}' for j in LABELS]))
    print('Stage 5 -> 4:',cm[4,3],'Stage 5 -> 6:',cm[4,5],
          'Stage 6 -> 5:',cm[5,4],'Stage 6 -> 4:',cm[5,3])
    dist=pd.DataFrame({'stage':LABELS,'true_count':[int((y==i).sum()) for i in LABELS],
                       'pred_count':[int((p==i).sum()) for i in LABELS]})
    dist['pred_ratio']=dist.pred_count/len(p)
    dist.to_csv(out/'predicted_stage_counts.csv',index=False)
    exp_rows=[]
    for exp,ind in frame.groupby('experiment_id',sort=False).indices.items():
        yy=np.asarray(y)[ind];pp=np.asarray(p)[ind]
        present=sorted(set(yy))
        exp_rows.append({'experiment_id':exp,'n':len(ind),
                         'f1_present_labels':f1_score(yy,pp,labels=present,average='macro',zero_division=0),
                         'f1_all_six_labels':score(yy,pp),
                         'accuracy':accuracy_score(yy,pp),
                         'tr_method':frame.iloc[ind[0]].get('tr_method',np.nan),
                         'stage5_support':int((yy==5).sum()),'stage5_pred4':int(((yy==5)&(pp==4)).sum()),
                         'stage6_support':int((yy==6).sum()),'stage6_pred5':int(((yy==6)&(pp==5)).sum())})
    experiments=pd.DataFrame(exp_rows).sort_values('f1_present_labels')
    experiments.to_csv(out/'experiment_metrics.csv',index=False)
    print('\nWorst experiments (present-label F1):')
    print(experiments[['experiment_id','n','f1_present_labels','tr_method']].head(6).to_string(index=False))
    for method,ids in frame.groupby('tr_method',sort=False).indices.items():
        print(f'tr_method={method}: Macro F1={score(np.asarray(y)[ids],np.asarray(p)[ids]):.4f} | N={len(ids)}')
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':score(y,p),
            'per_class_f1':{str(l):float(f) for l,f in zip(LABELS,
             precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0)[2])},
            '5_to_4':int(cm[4,3]),'5_to_6':int(cm[4,5]),'6_to_5':int(cm[5,4]),
            '6_to_4':int(cm[5,3])}

def run_experiment(b,group_list,output_dir,folds=5):
    name='B0_baseline' if not group_list else 'B_plus_'+'_'.join(group_list)
    print('\n'+'='*84+'\nEXPERIMENT',name,'\n'+'='*84)
    output_dir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    y=b['y']; g=b['exp_groups']; orig=b['orig']
    X_const=None if 'train_quantiles' in group_list else form_X(b,group_list)
    Xval_const=None if 'train_quantiles' in group_list else form_X(b,group_list,is_validation=True)
    original_train=b['train']
    gkf=GroupKFold(n_splits=folds)
    fold_results=[];oof=np.zeros((len(original_train),6),dtype=float)
    seen_oof=np.zeros(len(original_train),dtype=bool)
    n=len(original_train)
    for fid,(tr_idx,eval_idx) in enumerate(gkf.split(b['Xbase'],y,groups=g),1):
        if 'train_quantiles' in group_list:
            fit_orig_idx=tr_idx[orig[tr_idx]]
            thresholds=F.fit_quantile_thresholds(original_train.iloc[fit_orig_idx])
            X=form_X(b,group_list,threshold_spec=thresholds)
        else:X=X_const
        model=B.build_model(class_weight='balanced',algo='lgbm')
        model.fit(X.iloc[tr_idx],y.iloc[tr_idx])
        orig_idx=eval_idx[orig[eval_idx]]
        proba=model.predict_proba(X.iloc[orig_idx]);pred=np.asarray(model.classes_)[proba.argmax(axis=1)]
        f=score(y.iloc[orig_idx],pred);fold_results.append(float(f))
        for ci,cls in enumerate(model.classes_):oof[orig_idx,int(cls)-1]=proba[:,ci]
        seen_oof[orig_idx]=True
        print(f'  Fold {fid}: Macro F1={f:.4f}, original_val_n={len(orig_idx)}')
    print(f'[CV] {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    if not seen_oof.all():raise AssertionError('OOF incomplete')
    if 'train_quantiles' in group_list:
        threshold_spec=F.fit_quantile_thresholds(original_train)
        X=form_X(b,group_list,threshold_spec=threshold_spec)
        Xval=form_X(b,group_list,threshold_spec=threshold_spec,is_validation=True)
    else:
        threshold_spec=None;X=X_const;Xval=Xval_const
    model=B.build_model(class_weight='balanced',algo='lgbm')
    model.fit(X,y)
    proba=model.predict_proba(Xval)
    pred=np.asarray(model.classes_)[proba.argmax(axis=1)]
    proba_aligned=np.zeros((len(proba),6),dtype=float)
    for ci,cls in enumerate(model.classes_):proba_aligned[:,int(cls)-1]=proba[:,ci]
    yval=pd.to_numeric(b['val']['label'],errors='raise').astype(int).to_numpy()
    stats=evaluate_details(yval,pred,b['val'],name,output_dir)
    top=pd.Series(model.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
    print('\nTop 20 feature importances (split counts; not causal):\n'+top.head(20).to_string())
    top.rename('split_importance').to_csv(output_dir/'feature_importance.csv')
    if group_list:
        newcols=[c for c in X.columns if c.startswith('phys_')]
        print('[NEW FEATURES] split importance:',float(top.reindex(newcols).sum()),
              'of',float(top.sum()),'total (interpret cautiously)')
    elapsed=time.perf_counter()-start
    result={'experiment':name,'groups':list(group_list),'cv_mean':float(np.mean(fold_results)),
            'cv_std':float(np.std(fold_results)),'cv_folds':fold_results,
            'n_features':int(X.shape[1]),'n_train_augmented':len(X),
            'n_val':len(Xval),'seconds':round(elapsed,2),**stats}
    bundle={'model':model,'features':list(X.columns),'spec':b['spec'],'groups':list(group_list),
            'train_thresholds':threshold_spec,'class_weight':'balanced','postprocess':None,
            'augment_steps':[2,3],'origin':'teammate_v3B_plus_causal_physics_v1',
            'spatial_version':F.SPATIAL_VERSION if 'image_spatial' in group_list else None}
    joblib.dump(bundle,output_dir/'model.joblib')
    pd.DataFrame({'sample_id':b['val']['sample_id'],'label':yval,'pred':pred,
                  **{f'p{k}':proba_aligned[:,i] for i,k in enumerate(LABELS)}}).to_csv(output_dir/'validation_predictions.csv',index=False)
    pd.DataFrame({'sample_id':b['train']['sample_id'],'experiment_id':b['train']['experiment_id'],
                  'label':y.iloc[:n].to_numpy(),
                  **{f'p{k}':oof[:,i] for i,k in enumerate(LABELS)}}).to_csv(output_dir/'oof_train.csv',index=False)
    (output_dir/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result

def groups_arg(arg):
    if arg.strip().lower() in ('','none','baseline'):return []
    v=[x.strip() for x in arg.split(',') if x.strip()]
    if len(v)!=len(set(v)):raise ValueError('Duplicate groups')
    bad=set(v)-set(F.GROUPS)
    if bad:raise ValueError(f'Unknown groups: {bad}; choices={F.GROUPS}')
    return [x for x in F.GROUPS if x in v]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--image-cache',type=Path,default=Path('cache/v3b_cnn'))
    ap.add_argument('--spatial-cache',type=Path,default=Path('cache/v3b_physics'))
    ap.add_argument('--output-dir',type=Path,default=Path('artifacts/v3b_physics'))
    ap.add_argument('--groups',default='thermal_rates',help='comma separated, or none')
    ap.add_argument('--sweep',action='store_true',help='baseline + one-group-at-a-time CV and validation')
    ap.add_argument('--sweep-groups',nargs='+',default=['thermal_rates','peaks','electric_pressure','gas'])
    ap.add_argument('--folds',type=int,default=5)
    args=ap.parse_args()
    if args.folds<2:raise ValueError('folds >= 2')
    runs=[]
    if args.sweep:
        gs=args.sweep_groups
        invalid=set(gs)-set(F.GROUPS)
        if invalid:raise ValueError(f'Invalid sweep group(s): {invalid}')
        runs=[[]]+[[g] for g in gs]
    else:runs=[groups_arg(args.groups)]
    union=set().union(*map(set,runs))
    b=build_parts(args.data_dir,args.image_cache,args.spatial_cache,union)
    print(f'[DATA] train_original={len(b["train"])} train_aug={len(b["Xbase"])} validation={len(b["val"])}')
    rows=[]
    for groups in runs:
        out=args.output_dir/('B0_baseline' if not groups else 'B_plus_'+'_'.join(groups))
        result=run_experiment(b,groups,out,args.folds)
        rows.append({k:v for k,v in result.items() if not isinstance(v,(dict,list))})
        pd.DataFrame(rows).to_csv(args.output_dir/'comparison.csv',index=False)
    print('\n=== EXPERIMENT COMPARISON (select by TRAIN CV, validation as external check) ===')
    print(pd.DataFrame(rows)[['experiment','n_features','cv_mean','cv_std','macro_f1','5_to_4','5_to_6','6_to_5','seconds']].to_string(index=False))
    print('[SAVED]',args.output_dir)

if __name__=='__main__':main()
