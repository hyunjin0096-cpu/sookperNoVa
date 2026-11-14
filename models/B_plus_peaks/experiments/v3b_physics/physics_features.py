"""Causal physics-informed feature groups for the unchanged teammate V3-B model.

All features depend only on data in the supplied frame. Never use labels,
future rows, end-of-experiment statistics, or validation statistics.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
import model as B

GROUPS = ('thermal_rates', 'peaks', 'electric_pressure', 'gas', 'image_spatial', 'train_quantiles')
QUANTILES = (0.75, 0.90)
SPATIAL_NAMES = (
    'sp_hot_area', 'sp_hot_x', 'sp_hot_y', 'sp_hot_xspread', 'sp_hot_yspread',
    'sp_hot_center_dist', 'sp_hot_concentration', 'sp_heat_q1', 'sp_heat_q2',
    'sp_heat_q3', 'sp_heat_q4', 'sp_heat_left_right', 'sp_heat_top_bottom',
)
SPATIAL_VERSION = 'spatial_v1_heat_palette_full_image_128'


def safe_arr(frame, col):
    if col not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)


def split_indices(frame):
    """Original experiment ID grouping within THIS frame; current and past only."""
    t = safe_arr(frame, 'relative_time_s')
    e = frame['experiment_id'].astype(str).to_numpy()
    for group in pd.Series(np.arange(len(frame))).groupby(e, sort=False):
        idx = group[1].to_numpy()
        idx = idx[np.lexsort((frame['sample_id'].astype(str).to_numpy()[idx], t[idx]))]
        tt = t[idx]
        if not np.all(np.isfinite(tt)) or np.any(np.diff(tt) <= 0):
            raise ValueError(f'Invalid or duplicate relative_time_s in experiment {group[0]}')
        yield idx, tt


def history(t, x):
    """Returns instantaneous slope, acceleration, peak, peak-age, drawdown, cooling-age, signflip."""
    n=len(x)
    slope=np.full(n,np.nan); accel=np.full(n,np.nan)
    peak=np.full(n,np.nan); age=np.full(n,np.nan); gap=np.full(n,np.nan)
    cooling=np.zeros(n,dtype=float); flip=np.full(n,np.nan); new_peak=np.full(n,np.nan)
    high=-np.inf; peak_time=np.nan
    for j in range(n):
        if np.isfinite(x[j]):
            if j>0 and np.isfinite(x[j-1]):
                dt=t[j]-t[j-1]
                slope[j]=(x[j]-x[j-1])/dt
                if j>1 and np.isfinite(slope[j-1]):
                    accel[j]=(slope[j]-slope[j-1])/dt
                    flip[j]=float(np.sign(slope[j]) != np.sign(slope[j-1]) and
                                  slope[j] != 0 and slope[j-1] != 0)
                if slope[j] < 0:
                    cooling[j]=(cooling[j-1] if j>0 and np.isfinite(slope[j-1]) and slope[j-1]<0 else 0)+dt
            if x[j] > high:
                high=x[j]; peak_time=t[j]; new_peak[j]=1.0
            else:
                new_peak[j]=0.0
            peak[j]=high; gap[j]=high-x[j]; age[j]=t[j]-peak_time
        else:
            cooling[j]=np.nan
    return {'slope':slope,'accel':accel,'peak':peak,'age':age,'gap':gap,
            'cooling':cooling,'signflip':flip,'newpeak':new_peak}


def _window_std(t, x, w):
    out=np.full(len(x),np.nan)
    for j in range(len(x)):
        first=np.searchsorted(t, t[j]-w, side='right')
        z=x[first:j+1]; z=z[np.isfinite(z)]
        if len(z):out[j]=float(np.std(z))
    return out


def fit_quantile_thresholds(original_train_frame):
    """Fit thresholds on *original TRAIN-fold* observations, never validation/test."""
    result={}
    for col in ('meas_t1','meas_t2'):
        vals=safe_arr(original_train_frame, col)
        vals=vals[np.isfinite(vals)]
        if not len(vals): continue
        result[col]=[float(v) for v in np.quantile(vals, QUANTILES)]
    return result


def _threshold_series(t, x, val):
    above=np.full(len(x),np.nan); after=np.full(len(x),np.nan); dwelling=np.full(len(x),np.nan)
    first=np.nan; duration=0.0
    for j in range(len(x)):
        if not np.isfinite(x[j]):continue
        flag=x[j]>=val
        above[j]=float(flag)
        if flag and not np.isfinite(first):first=t[j]
        after[j]=(t[j]-first) if np.isfinite(first) else np.nan
        if j>0 and flag and np.isfinite(x[j-1]) and x[j-1]>=val:
            duration+=t[j]-t[j-1]
        dwelling[j]=duration
    return above,after,dwelling


def spatial_stats(path):
    """Relative false-colour spatial statistics; NOT calibrated temperatures."""
    with Image.open(path) as im:
        im=im.convert('RGB'); im.thumbnail((128,128))
        h,_,_=B._heat_index(im)
    h=np.asarray(h,dtype=np.float32)
    hei,wid=h.shape
    hot=(h>=0.8)
    if hot.any():
        ys,xs=np.where(hot)
        xx=(xs+0.5)/wid; yy=(ys+0.5)/hei
        mx=float(xx.mean()); my=float(yy.mean()); sx=float(xx.std()); sy=float(yy.std())
        dist=float(np.hypot(mx-0.5,my-0.5))
        conc=float(1.0/(1.0+25.0*(sx*sx+sy*sy)))
    else:
        mx=my=sx=sy=dist=conc=np.nan
    hm=hei//2; wm=wid//2
    quad=[float(v.mean()) for v in (h[:hm,:wm],h[:hm,wm:],h[hm:,:wm],h[hm:,wm:])]
    return [float(hot.mean()),mx,my,sx,sy,dist,conc,*quad,
            float(h[:,:wm].mean()-h[:,wm:].mean()),
            float(h[:hm,:].mean()-h[hm:,:].mean())]


def extract_spatial(frame, paths, cache_path=None, n_jobs=2):
    """Cache by exact sample_id order and image feature version; no labels."""
    ids=frame['sample_id'].astype(str).to_numpy()
    cache_path=Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            t=pd.read_csv(cache_path,dtype={'sample_id':'string'}).set_index('sample_id')
            if list(t.columns)==list(SPATIAL_NAMES) and len(t)==len(ids) and set(t.index)==set(ids):
                print('[SPATIAL] cache reused:',cache_path)
                return t
        except Exception:pass
    rows=Parallel(n_jobs=n_jobs,batch_size=64)(delayed(spatial_stats)(p) for p in paths)
    out=pd.DataFrame(rows,index=pd.Index(ids,name='sample_id'),columns=SPATIAL_NAMES)
    if cache_path:
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        out.to_csv(cache_path)
        print('[SPATIAL] saved:',cache_path)
    return out


def physics_features(frame, groups, spatial=None, thresholds=None):
    groups=set(groups)
    invalid=groups-set(GROUPS)
    if invalid:raise ValueError(f'Unknown feature groups: {invalid}')
    n=len(frame)
    if n==0:return pd.DataFrame(index=frame.index)
    vals={c:safe_arr(frame,c) for c in (
        'meas_t1','meas_t2','thermal_t','meas_v','meas_p',*B.GAS_COLS)}
    out={}
    for name in groups:
        if name=='train_quantiles' and not thresholds:raise ValueError('train_quantiles needs Train-fold thresholds')
        if name=='image_spatial' and spatial is None:raise ValueError('image_spatial needs image features')
    if 'image_spatial' in groups:
        image=spatial.reindex(frame['sample_id'].astype(str).to_numpy())
        if image.isna().all(axis=1).any():raise ValueError('Spatial features missing for sample_ids')
        for c in SPATIAL_NAMES:out['phys_'+c]=pd.to_numeric(image[c],errors='coerce').to_numpy()
    fields={}
    def field(key):
        if key not in fields: fields[key]=np.full(n,np.nan)
        return fields[key]
    for idx,t in split_indices(frame):
        local={k:v[idx] for k,v in vals.items()}
        needs=set()
        if groups.intersection(('thermal_rates','peaks')):needs.update(('meas_t1','meas_t2'))
        if 'electric_pressure' in groups:needs.update(('meas_v','meas_p','meas_t1','meas_t2'))
        if 'gas' in groups:needs.update(B.GAS_COLS)
        hs={k:history(t,local[k]) for k in needs}
        if 'thermal_rates' in groups:
            for c in ('meas_t1','meas_t2'):
                key='T1' if c=='meas_t1' else 'T2'; h=hs[c]
                field(f'phys_d2{key}_dt2')[idx]=h['accel']
                field(f'phys_{key}_slope_signflip')[idx]=h['signflip']
                field(f'phys_{key}_cooling_duration_s')[idx]=h['cooling']
                field(f'phys_{key}_new_peak')[idx]=h['newpeak']
                field(f'phys_{key}_std30s')[idx]=_window_std(t,local[c],30)
            field('phys_T1_T2_slope_diff')[idx]=hs['meas_t1']['slope']-hs['meas_t2']['slope']
            both=np.isfinite(hs['meas_t1']['slope']) & np.isfinite(hs['meas_t2']['slope'])
            opp=(hs['meas_t1']['slope']*hs['meas_t2']['slope']<0).astype(float)
            field('phys_T1_T2_opposite_slope')[idx]=np.where(both,opp,np.nan)
        if 'peaks' in groups:
            for c in ('meas_t1','meas_t2'):
                key='T1' if c=='meas_t1' else 'T2'; h=hs[c]
                # Original B has *_cumext; these are new GAP/PEAK-AGE representations.
                field(f'phys_{key}_peak_gap')[idx]=h['gap']
                field(f'phys_{key}_time_since_peak_s')[idx]=h['age']
            field('phys_T1_T2_peak_gap_diff')[idx]=hs['meas_t1']['gap']-hs['meas_t2']['gap']
        if 'electric_pressure' in groups:
            v=hs['meas_v']; p=hs['meas_p']; a=hs['meas_t1'];b=hs['meas_t2']
            field('phys_d2P_dt2')[idx]=p['accel']
            field('phys_d2V_dt2')[idx]=v['accel']
            field('phys_P_std30s')[idx]=_window_std(t,local['meas_p'],30)
            field('phys_V_std30s')[idx]=_window_std(t,local['meas_v'],30)
            vs=v['slope'];ps=p['slope'];ts=a['slope'];t2=b['slope']
            field('phys_vdrop_t1rise_joint')[idx]=(-np.minimum(vs,0))*np.maximum(ts,0)
            field('phys_vdrop_t2rise_joint')[idx]=(-np.minimum(vs,0))*np.maximum(t2,0)
            field('phys_pressure_t1rise_joint')[idx]=np.maximum(ps,0)*np.maximum(ts,0)
            field('phys_pressure_t2rise_joint')[idx]=np.maximum(ps,0)*np.maximum(t2,0)
        if 'gas' in groups:
            arr=np.stack([local[c] for c in B.GAS_COLS],axis=1)
            pos=np.where(np.isfinite(arr),arr>0,np.nan)
            positive=np.nansum(pos,axis=1)
            known=np.isfinite(arr).sum(axis=1)
            field('phys_gas_positive_sensor_count')[idx]=np.where(known>0,positive,np.nan)
            field('phys_gas_valid_sensor_count')[idx]=known.astype(float)
            # This is a sensor-reading indicator, NOT proof of gas release.
            gaslog=np.log1p(np.clip(arr,0,None))
            gaslog[~np.isfinite(arr)]=np.nan
            group_a=[B.GAS_COLS.index(c) for c in ('meas_hcn','meas_hf','meas_n2o','meas_no','meas_no2')]
            group_b=[B.GAS_COLS.index(c) for c in ('meas_ch4','meas_co')]
            def sum_or_nan(a):
                z=np.nansum(a,axis=1).astype(float)
                z[np.isfinite(a).sum(axis=1)==0]=np.nan
                return z
            a=sum_or_nan(gaslog[:,group_a]);b=sum_or_nan(gaslog[:,group_b]);total=sum_or_nan(gaslog)
            field('phys_gas_loggroup_a')[idx]=a;field('phys_gas_loggroup_b')[idx]=b
            field('phys_gas_loggroup_diff')[idx]=a-b
            field('phys_gas_logsum')[idx]=total
            good=np.isfinite(a)&np.isfinite(b)
            field('phys_gas_group_coactive')[idx]=np.where(good,((a>0)&(b>0)).astype(float),np.nan)
            g=history(t,total)
            field('phys_gas_logsum_slope_prev')[idx]=g['slope']
            field('phys_gas_logsum_accel')[idx]=g['accel']
            field('phys_gas_logsum_std30s')[idx]=_window_std(t,total,30)
            # Rate of group-level gas readings, not absolute gas generation rates.
            field('phys_gas_group_a_slope')[idx]=history(t,a)['slope']
            field('phys_gas_group_b_slope')[idx]=history(t,b)['slope']
        if 'image_spatial' in groups:
            # spatial values for current frame and past frames only
            x=out['phys_sp_hot_area'][idx]
            hh=history(t,x)
            field('phys_sp_hot_area_slope_prev')[idx]=hh['slope']
            field('phys_sp_hot_area_accel')[idx]=hh['accel']
            for w in (30,60):
                _,_,delta,slope=B._window_stats(t,x,float(w))
                field(f'phys_sp_hot_area_delta{w}s')[idx]=delta
                field(f'phys_sp_hot_area_slope{w}s')[idx]=slope
        if 'train_quantiles' in groups:
            for c in ('meas_t1','meas_t2'):
                if c not in thresholds:continue
                for qi,thr in enumerate(thresholds[c]):
                    above,age,duration=_threshold_series(t,local[c],thr)
                    key=f'{c}_q{int(QUANTILES[qi]*100)}'
                    field(f'phys_{key}_above')[idx]=above
                    field(f'phys_{key}_age_s')[idx]=age
                    field(f'phys_{key}_duration_s')[idx]=duration
    out.update(fields)
    df=pd.DataFrame(out,index=frame.index)
    if len(set(df.columns))!=len(df.columns):raise AssertionError('Duplicate physics features')
    return df.replace([np.inf,-np.inf],np.nan).astype(np.float32)
