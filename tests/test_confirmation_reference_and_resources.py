import os
import psutil
import pytest
from scarx.evaluation.r8_confirmation import cluster_statistics
from scarx.evaluation.forensic.confirmation_reference import reference_statistics
from scarx.evaluation.confirmation_resources import ConfirmationResources


@pytest.mark.parametrize('effects',[[.1]*12,[0.]*12,[(-1)**i*(i+1)/20 for i in range(12)]])
def test_scalar_reference_agrees_without_using_fresh_seed_models(effects):
    a=cluster_statistics(effects);b=reference_statistics(effects)
    for field in b:assert a[field]==pytest.approx(b[field],abs=1e-12)


def test_observer_pid_equal_to_root_is_not_double_counted():
    current=psutil.Process(os.getpid());m=ConfirmationResources(observer_identity=(current.pid,current.create_time()))
    m.sample();report=m.report() if hasattr(m,'start') else None
    assert m.observed_pids=={current.pid} and m.peak_rss>0 and not m.measurement_errors


def test_missing_service_is_not_an_unrelated_reused_pid():
    current=psutil.Process(os.getpid())
    # Same PID with deliberately different creation identity is not adopted.
    m=ConfirmationResources(observer_identity=(current.pid,current.create_time()-10));m.sample()
    assert m.observed_pids=={current.pid} and not m.measurement_errors
