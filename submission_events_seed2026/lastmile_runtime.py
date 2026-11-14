"""Causal feature additions; thresholds are fixed before inspecting Test."""
import numpy as np
import pandas as pd
from features import fuse
from train_frozen_cv import add_gas_status,add_history

EXTRA=['thermal_t','indu_t','tr_method','bat_class','bat_cc','soc_status',
       'meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf',
       'meas_n2o','meas_no','meas_no2','meas_so2']
SIGNALS=['meas_v','meas_t1','meas_t2','meas_p','thermal_t']


def event_features(frame):
    """42 current/past-only fields. No labels, end times or fitted statistics.

    Baseline median uses observations in [first_time, first_time+30s] that have
    ALREADY arrived. It stabilizes after 30 seconds; never backfills earlier rows.
    """
    frame=frame.reset_index(drop=True)
    output={}
    def place(name,ids,values):
        if name not in output: output[name]=np.full(len(frame),np.nan,dtype=float)
        output[name][ids]=values
    for ids in frame.groupby('experiment_id',sort=False).indices.values():
        ids=np.asarray(ids)
        time=pd.to_numeric(frame.iloc[ids].relative_time_s,errors='coerce').to_numpy(float)
        order=np.argsort(time,kind='stable');ids=ids[order];time=time[order]
        if not np.isfinite(time).all() or (np.diff(time)<=0).any(): raise ValueError('Invalid event timestamps')
        g=frame.iloc[ids]
        signal_values={}
        baselines={}
        peaks={}
        for col in SIGNALS:
            v=pd.to_numeric(g[col],errors='coerce').to_numpy(float)
            v[~np.isfinite(v)]=np.nan
            signal_values[col]=v
            base=np.full(len(v),np.nan)
            early=np.flatnonzero(time<=time[0]+30.)
            for i in early:
                valid=v[:i+1][np.isfinite(v[:i+1])]
                if len(valid): base[i]=np.median(valid)
            if len(early): base[len(early):]=base[early[-1]]
            baselines[col]=base
            maximum=np.full(len(v),np.nan);age=np.full(len(v),np.nan)
            best=-np.inf;best_time=None
            for i,value in enumerate(v):
                if np.isfinite(value) and value>best:
                    best=value;best_time=time[i]
                if best_time is not None:
                    maximum[i]=best;age[i]=time[i]-best_time
            peaks[col]=maximum
            place('evt_'+col+'_baseline_delta',ids,v-base)
            place('evt_'+col+'_baseline_relative_delta',ids,(v-base)/np.maximum(np.abs(base),1.))
            place('evt_'+col+'_since_strict_peak_s',ids,age)
        def first_event(name,hit):
            first=np.minimum.accumulate(np.where(hit,np.arange(len(hit)),len(hit)))
            seen=first<len(hit)
            age=np.full(len(hit),np.nan)
            age[seen]=time[seen]-time[first[seen]]
            place('evt_'+name+'_seen',ids,seen.astype(float))
            place('evt_'+name+'_age_s',ids,age)
        for col in ['meas_t1','meas_t2','thermal_t']:
            delta=signal_values[col]-baselines[col]
            for threshold in [10,30,60]:
                first_event(f'{col}_rise_{threshold}',delta>=threshold)
        voltage=signal_values['meas_v'];vmax=peaks['meas_v']
        ratio=np.divide(voltage,vmax,out=np.full(len(voltage),np.nan),where=np.isfinite(vmax)&(vmax>1e-6))
        place('evt_voltage_to_past_peak',ids,ratio)
        for ratio_threshold in [.95,.8,.5,.1]:
            first_event(f'voltage_below_{int(ratio_threshold*100)}pct_peak',ratio<=ratio_threshold)
    return pd.DataFrame(output,index=np.arange(len(frame))).replace([np.inf,-np.inf],np.nan).astype('float32')


def build_features(frame,embeddings,stats,variant):
    frame=frame.reset_index(drop=True)
    x=add_history(add_gas_status(fuse(frame,embeddings,stats,EXTRA),frame),frame)
    if variant=='events': x=pd.concat([x,event_features(frame)],axis=1)
    elif variant=='no_time': x=x.drop(columns=['relative_time_s'])
    elif variant not in ['history','method_history']: raise ValueError(variant)
    if x.columns.duplicated().any(): raise ValueError('Duplicate feature names')
    return x.astype('float32')
