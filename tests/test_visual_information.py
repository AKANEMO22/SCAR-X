import numpy as np
import pytest
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.visual_information import observe,protected,ridge,collision,counterfactual_pairs,source_model
from scarx.evaluation.vision_fixtures import scenes,render


def test_readonly_path_matches_actual_bos_input_and_rejects_labels_without_updates():
    m=VisualWords(['<eos>','<bos>','red','green','circle'],33113);image=render(scenes()[0]);m.calibrate([{'prompt':'What color?','image':image}])
    original=m.dumps();copy=VisualWords.loads(original);before=protected(m);r=observe(m,'What color?',image)
    copy.prepare('What color?',image);f=copy.word_feature(1,0)
    np.testing.assert_array_equal(r['vectors']['bos_feature'],f);np.testing.assert_array_equal(r['vectors']['head_feature'],copy.head.transform(f))
    assert r['ticks']==36 and r['first_token']==copy.head.probabilities(f,copy.output_mask).argmax() and protected(m)==before
    after=m.dumps()
    with pytest.raises(TypeError):observe(m,'What color?',image,target='red')
    assert m.dumps()==after
    with pytest.raises(ValueError):source_model(original,control=True)


def test_fixed_ridge_matches_expanded_exposures_excludes_unseen_rows_and_shuffles_only_labels():
    train=np.array([[0.,0.],[1.,0.],[0.,1.],[100.,100.]]);dev=np.array([[.1,0.],[.9,0.]]);labels=np.array([0,1,0,1]);exposures=np.array([0,1,1,2,2,2])
    a=ridge(train,dev,labels,2,exposures);b=ridge(train[exposures],dev,labels[exposures],2,np.arange(6))
    np.testing.assert_array_equal(a['dev'],b['dev']);changed=train.copy();changed[3]=-10000
    np.testing.assert_array_equal(a['dev'],ridge(changed,dev,labels,2,exposures)['dev'])
    shuffled=ridge(train,dev,labels,2,exposures,shuffle_seed=9)
    assert shuffled['training_label_histogram']==a['training_label_histogram'] and a['exposure_counts'].tolist()==[1,2,3,0]


def test_exact_collision_ceiling_uses_equal_task_mass_and_pairs_do_not_select_same_labels():
    cases=[{'id':'a','kind':'single_color','language':'vi','scene_id':'s1','prompt':'color'},
        {'id':'b','kind':'single_color','language':'vi','scene_id':'s2','prompt':'color'},
        {'id':'c','kind':'single_shape','language':'vi','scene_id':'s1','prompt':'shape'}]
    r=collision(['same','same','other'],[0,1,2],cases)
    assert r['oracle_micro_ceiling']==pytest.approx(2/3) and r['oracle_macro_ceiling']==pytest.approx(.75) and r['conflicting_groups']==1
    pairs=counterfactual_pairs(cases,[0,1,2]);assert pairs['image']==[(0,1)] and pairs['question']==[(0,2)]
