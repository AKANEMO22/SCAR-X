import pytest
from scarx.evaluation.endurance_independent_audit import (
    original_long_gate,final_audit_decision,video_coverage,resource_coverage)


def test_final_receipt_and_real_worker_exit_are_required_even_after_eight_hours():
    flags=dict(terminal_receipt=False,worker_live=True,registered_gate=True,
        source_archive_valid=True,preflight_valid=True,sampled_reserve_valid=True,
        receipt_trace_private_agree=True,cleanup_valid=True)
    assert final_audit_decision(**flags)=='PRELIMINARY_LIVE_NOT_A_COMPLETED_PASS'
    flags.update(terminal_receipt=True)
    assert 'WAIT' in final_audit_decision(**flags)
    flags.update(worker_live=False)
    assert final_audit_decision(**flags)=='EIGHT_HOUR_ENGINEERING_DATA_AUDIT_PASS_ONLY'
    flags['receipt_trace_private_agree']=False
    assert 'NOT_QUALIFIED' in final_audit_decision(**flags)


def test_independent_gate_rejects_nonfinite_forged_counts_and_short_duration():
    base=dict(elapsed=28800,cycles=96,checkpoints=16,updates=100,replay_seen=20,
              device='gpu',critical_failures=0,active_capture=False)
    assert original_long_gate(**base)
    assert not original_long_gate(**(base|{'elapsed':240}))
    for bad in (float('nan'),float('inf'),-1,True):
        with pytest.raises(ValueError):original_long_gate(**(base|{'elapsed':bad}))
    with pytest.raises(ValueError):original_long_gate(**(base|{'cycles':80.0}))


def test_repeated_short_clips_do_not_become_hours_of_unique_continuous_video():
    def cycle(identifier):
        return {'stream':{'id':identifier,'duration':21.,'created':10.,'original':'owned-clip',
            'skipped_intervals':[],'asr_errors':[],'status':'complete','analyzed_intervals':[{}]},
            'actual_answer':{'created':42.,'evidence':[]},'answer_quality_scored':False}
    rows=[cycle('one'),cycle('two')]
    summary=video_coverage(rows)
    assert summary['summed_repeated_video_seconds']==42
    assert summary['unique_original_source_paths']==1
    assert not summary['continuous_video_hours_proven'] and not summary['semantic_answers_scored']
    with pytest.raises(ValueError):video_coverage([rows[0],rows[0]])


def test_resource_group_coverage_does_not_silently_include_a_new_gateway():
    beat={'resources':{'peak_rss_bytes':100,'peak_private_commit_bytes':150,
        'peak_gpu_dedicated_bytes':50,'peak_conservative_ram_bytes':120,
        'gpu_shared':{'bytes':20},'processes':[{'pid':7}],'included_client_pids':[7],'limit_bytes':None},
        'GPU_headroom':{'free_MiB':1024}}
    result=resource_coverage([beat],current_gateway_pid=9)
    assert result['peak_reported_tree_RSS_bytes']==100 and result['peak_reported_tree_private_commit_bytes']==150
    assert result['current_gateway_in_last_sample'] is False
    assert result['RSS_and_commit_not_added']
