"""Synthetic complete-feature causality test. No real data or weights."""
import numpy as np
import pandas as pd


def main():
    from features import fuse
    from frozen_features import GAS_COLUMNS,add_gas_status,add_history,extra_for_variant
    from baseline_features import IMAGE_STATS
    rows=[]
    for e in ['A','B']:
        for i,t in enumerate([0,5,15,30,65]):
            row={c:float(i) for c in GAS_COLUMNS}
            row.update(sample_id=f'{e}_{i}',experiment_id=e,relative_time_s=t,
                meas_v=4-i*.2,meas_a=0.,meas_t1=20+i*3,meas_t2=25+i,meas_p=2+i,
                thermal_t=30+i,indu_t=50+i,tr_method=1,bat_class=1,bat_cc=6.,soc_status=2)
            rows.append(row)
    frame=pd.DataFrame(rows)
    vectors=np.random.default_rng(42).normal(size=(len(frame),512)).astype('float32')
    image_stats=pd.DataFrame(np.random.default_rng(43).random((len(frame),6)),columns=IMAGE_STATS,index=frame.sample_id)
    extra=extra_for_variant(frame,['thermal_t','indu_t','tr_method','bat_class'],'conditions_gas_history')
    def make(g):
        indices=pd.Index(frame.sample_id).get_indexer(g.sample_id)
        emb=pd.DataFrame(vectors[indices],columns=[f'resnet_{i:03d}' for i in range(512)])
        return add_history(add_gas_status(fuse(g,emb,image_stats,extra),g),g)
    full=make(frame)
    assert full.shape==(10,607),full.shape
    shuffled=frame.sample(frac=1,random_state=7).reset_index(drop=True)
    expected=full.iloc[pd.Index(frame.sample_id).get_indexer(shuffled.sample_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(make(shuffled),expected)
    for _,g in frame.groupby('experiment_id'):
        for cutoff in [15,65]:
            part=g[g.relative_time_s<=cutoff].reset_index(drop=True)
            expected=full.iloc[pd.Index(frame.sample_id).get_indexer(part.sample_id)].reset_index(drop=True)
            pd.testing.assert_frame_equal(make(part),expected)
    labeled=frame.copy();labeled['label']=6
    pd.testing.assert_frame_equal(make(labeled),full)
    print('PASS: all 607 features, shuffled order, experiment isolation, prefix invariance, label independence')


if __name__=='__main__': main()
