"""Pure independent coverage/resource accounting; never a model-quality scorer."""

import math
import statistics


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError("Expected finite nonnegative measured number")
    return value


def count(value):
    if type(value) is not int or value<0:
        raise ValueError("Expected nonnegative measured count")
    return value


def original_long_gate(*, elapsed,cycles,checkpoints,updates,replay_seen,device,
                       critical_failures,active_capture):
    values=[count(v) for v in (cycles,checkpoints,updates,replay_seen,critical_failures)]
    number(elapsed)
    if type(active_capture) is not bool:
        raise ValueError("Expected active-capture observation")
    return bool(elapsed>=28800 and cycles>=80 and checkpoints>=8
        and updates>0 and updates>replay_seen and device=='gpu'
        and critical_failures==0 and not active_capture)


def final_audit_decision(*,terminal_receipt,worker_live,registered_gate,
                        source_archive_valid,preflight_valid,sampled_reserve_valid,
                        receipt_trace_private_agree,cleanup_valid):
    flags=(terminal_receipt,worker_live,registered_gate,source_archive_valid,
           preflight_valid,sampled_reserve_valid,receipt_trace_private_agree,cleanup_valid)
    if any(type(value) is not bool for value in flags):
        raise ValueError("Audit evidence flags must be explicit booleans")
    if not terminal_receipt:
        return 'PRELIMINARY_LIVE_NOT_A_COMPLETED_PASS' if worker_live else 'INTERRUPTED_OR_MISSING_RECEIPT_NOT_PASS'
    if worker_live:
        return 'FINAL_RECEIPT_PRESENT_WAIT_FOR_ACTUAL_WORKER_EXIT'
    return ('EIGHT_HOUR_ENGINEERING_DATA_AUDIT_PASS_ONLY' if all(flags[i] for i in (2,3,4,5,6,7))
            else 'FINAL_ENGINEERING_AUDIT_NOT_QUALIFIED')


def video_coverage(cycles):
    seen=set();total=0.;latencies=[];source_paths=set();skips=asr_failures=0
    per_cycle=[]
    for record in cycles:
        state,answer=record['stream'],record['actual_answer']
        identifier=state['id']
        if identifier in seen:
            raise ValueError("Duplicate completed video stream")
        seen.add(identifier)
        duration=number(state['duration']);total+=duration
        latency=number(answer['created'])-number(state['created'])
        if latency<0:raise ValueError("Answer preceded source observation")
        latencies.append(latency);source_paths.add(state['original'])
        skips+=len(state['skipped_intervals']);asr_failures+=len(state['asr_errors'])
        if record['answer_quality_scored'] is not False:
            raise ValueError("Original engineering video answers are unscored")
        per_cycle.append({'stream_id':identifier,'duration_seconds':duration,
                          'observation_to_answer_seconds':latency,'status':state['status'],
                          'analyzed_intervals':len(state['analyzed_intervals']),
                          'selected_evidence_count':len(answer.get('evidence',[]))})
    return {'completed_streams':len(seen),'summed_repeated_video_seconds':total,
        'unique_original_source_paths':len(source_paths),'original_paths':sorted(source_paths),
        'skipped_intervals':skips,'ASR_failures_in_stream_records':asr_failures,
        'median_observation_to_answer_seconds':statistics.median(latencies) if latencies else None,
        'first_five_latency_median':statistics.median(latencies[:5]) if latencies else None,
        'last_five_latency_median':statistics.median(latencies[-5:]) if latencies else None,
        'semantic_answers_scored':False,'continuous_video_hours_proven':False,'per_cycle':per_cycle}


def resource_coverage(beats,current_gateway_pid=None):
    rss=commit=dedicated=shared=ram=0
    free=[];clients=set();last_pids=set();start_pids=set()
    for index,beat in enumerate(beats):
        resource=beat['resources']
        rss=max(rss,number(resource['peak_rss_bytes']))
        commit=max(commit,number(resource['peak_private_commit_bytes']))
        dedicated=max(dedicated,number(resource['peak_gpu_dedicated_bytes']))
        ram=max(ram,number(resource['peak_conservative_ram_bytes']))
        shared=max(shared,number(resource['gpu_shared']['bytes']))
        free.append(number(beat['GPU_headroom']['free_MiB']))
        last_pids={count(p['pid']) for p in resource['processes']}
        if index==0:start_pids=last_pids.copy()
        clients.update(count(pid) for pid in resource.get('included_client_pids',[]))
        if resource['limit_bytes'] is not None:
            raise ValueError("GPU endurance was registered without a host RAM cap")
    return {'heartbeat_samples':len(beats),'peak_reported_tree_RSS_bytes':rss,
        'peak_reported_tree_private_commit_bytes':commit,'peak_reported_tree_dedicated_GPU_bytes':dedicated,
        'peak_reported_conservative_RAM_bytes':ram,'max_heartbeat_shared_GPU_bytes':shared,
        'sampled_min_global_free_VRAM_MiB':min(free) if free else None,
        'sampled_512MiB_reserve_met':bool(free) and min(free)>=512,
        'initial_resource_PIDs':sorted(start_pids),'last_resource_PIDs':sorted(last_pids),
        'initial_client_roots_used_by_fixed_monitor':sorted(clients),
        'validated_current_Hermes_gateway_pid':current_gateway_pid,
        'current_gateway_in_last_sample':current_gateway_pid in last_pids if current_gateway_pid else None,
        'RSS_and_commit_not_added':True,'GPU_mode_host_RAM_uncapped':True,
        'scope':'Recorded process trees/client roots; not a proven complete current-Hermes/OS peak. Physical free is sampled, not an unobserved allocation guarantee.'}
