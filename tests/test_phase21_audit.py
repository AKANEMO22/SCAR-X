import ast
from pathlib import Path
import numpy as np
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.phase21 import holm,signflip


def test_strict_binding_has_no_value_count_shortcut():
    x,y=strict_binding(99,256)
    for channel in range(4):
        assert np.all((x[:,:,channel]>1).sum(1)==1)
    assert y.sum()==128


def test_all_neural_and_learning_modules_have_no_backward_dependency():
    root=Path(__file__).resolve().parents[1]/'src/scarx'
    for domain in ('neural','learning'):
        for path in (root/domain).glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    assert not any(a.name.startswith(('torch','jax','tensorflow')) for a in node.names)
                if isinstance(node,ast.ImportFrom):
                    assert not (node.module or '').startswith(('torch','jax','tensorflow'))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    assert node.func.attr not in ('backward','autograd')


def test_seed_level_exact_statistics():
    assert signflip([1.]*8)==2/256
    assert signflip([0.]*8)==1.
    assert holm([.01,.04,.03])==[.03,.06,.06]


def test_strict_binding_label_is_recoverable_from_key_value_relation():
    x,y=strict_binding(109,256)
    predicted=[]
    for seq in x:
        query=int(np.argmax(seq[10,4:6]))
        event=int(np.argmax(seq[:10,query]))
        predicted.append(int(np.argmax(seq[event,2:4])))
    np.testing.assert_array_equal(predicted,y)
