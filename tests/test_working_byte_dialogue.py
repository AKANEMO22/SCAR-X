import numpy as np
from scarx.evaluation.working_byte_dialogue import WorkingByteDialogue


def test_snapshot_preserves_same_core_activation_without_hidden_weight_updates():
    m=WorkingByteDialogue(9103,'snapshot_normalized');row={'id':'a','messages':[{'role':'user','text':'hi'}],'target':'Hello.'}
    m.calibrate([row]);wi=m.core.w_in.copy();wr=m.core.w_rec.copy()
    m.learn(row['messages'],row['target']);np.testing.assert_array_equal(m.core.w_in,wi);np.testing.assert_array_equal(m.core.w_rec,wr)
    context=m._prepare(row['messages']);held=m.held.copy()
    for i in range(48):m.backbone.byte_step(ord('a')+2,context,i)
    np.testing.assert_array_equal(m.held,held)
    clone=WorkingByteDialogue.loads(m.dumps());assert clone.dumps()==m.dumps()
    assert m.generate(row['messages'])==clone.generate(row['messages'])
    saved=m.dumps();m.score([row],'zero');m.score([row],'swap');assert m.dumps()==saved
    assert m.head.weights.shape==(268,96) and m.core.config.hidden==48
