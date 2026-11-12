from scarx.evaluation.forensic.r9_review import effective_params


def test_unused_gate_thresholds_and_disabled_cascade_do_not_create_new_models():
    a={'gating_mode':'none','sensory_threshold':.4,'sensory_floor':.15,'use_bf_cascade':False,'kappa1':.004,'kappa2':.001}
    b={**a,'sensory_threshold':1.,'kappa1':.02}
    assert effective_params(a)==effective_params(b)


def test_active_parameters_and_reset_modes_remain_distinct():
    a={'reset_mode':'damped_subtractive','damped_lambda':.2,'gating_mode':'hard_sensory','sensory_threshold':.6}
    assert effective_params(a)!=effective_params({**a,'damped_lambda':.8})
    assert effective_params(a)!=effective_params({**a,'sensory_threshold':1.})


def test_fast_sigmoid_width_is_ignored_but_gaussian_width_is_not():
    a={'surrogate_fn':'fast_sigmoid','surrogate_param':.3}
    assert effective_params(a)==effective_params({**a,'surrogate_param':.8})
    b={**a,'surrogate_fn':'gaussian'}
    assert effective_params(b)!=effective_params({**b,'surrogate_param':.8})
