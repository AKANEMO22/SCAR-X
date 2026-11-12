import numpy as np
from scarx.evaluation.acr_study import build_data,stress_cases,TASKS,SEEDS,means
from scarx.evaluation.r8_matched_study import event_oracle,refit_predict

def test_joint_budget_no_task_channel_and_single_pooled_refit():
    bank=build_data(SEEDS[0]);joint=bank['joint']
    assert joint['train'][0].shape==(1024,12,8)
    assert joint['refit'][0].shape==(256,12,8)
    assert joint['id'][0].shape==(2048,12,8)
    assert np.bincount(joint['train_families']).tolist()==[256]*4
    assert np.bincount(joint['refit_families']).tolist()==[64]*4
    # Provenance lives only in evaluator; data entering learner remains8channels.
    for i,t in enumerate(TASKS):
        mask=joint['id_families']==i
        np.testing.assert_array_equal(joint['id'][0][mask],bank[t]['id'][0])

def test_stress_marker_protected_query_counterfactual_and_delay():
    bank=build_data(SEEDS[0])
    for t in TASKS:
        d=bank[t];cases=stress_cases(t,d)
        for name,(x,y) in cases.items():
            np.testing.assert_array_equal(x[:,-1],d['id'][0][:128,-1])
            assert x.shape[2]==8 and len(y)==128
            if name!='query_flip':np.testing.assert_array_equal(y,d['id'][1][:128])
        assert cases['delay12'][0].shape==(128,24,8)
        if 'query_flip' in cases:
            labels,_=event_oracle(t,cases['query_flip'][0])
            np.testing.assert_array_equal(labels,cases['query_flip'][1])
            np.testing.assert_array_equal(labels,1-d['id'][1][:128])

def test_means_does_not_pool_signal_shuffle_with_candidate():
    rows=[]
    for p in ('static_50','readout_only','state_signal_shuffle'):
        for t in TASKS+('joint',):
            for seed in SEEDS:
                for condition in ('learned_mtsg','readout_only','shuffled_signal'):
                    rows.append({'profile':p,'task':t,'seed':seed,'condition':condition,'budget':'equal_updates',
                        'metrics':{k:(.7 if condition=='learned_mtsg' else .2 if condition=='readout_only' else .1) for k in ('id_direct','id_probe','ood_direct','ood_probe')}})
    m=means(rows,('static_50','readout_only','state_signal_shuffle'))
    assert np.isclose(m['static_50']['xor']['id_direct'],.7)
    assert np.isclose(m['readout_only']['xor']['id_direct'],.2)
    assert np.isclose(m['state_signal_shuffle']['xor']['id_direct'],.1)
