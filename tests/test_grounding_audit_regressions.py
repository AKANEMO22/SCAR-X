"""Regression tests for the data-grounding audit mistakes actually made.

Each test locks in a confusion that occurred during the audit, so it cannot
silently recur:

  1. 386 CONTENT tokens vs 388 VOCABULARY slots (the 2 control tokens)
  2. intersection vs naive subtraction when counting tokens
  3. question-only baseline mistaken for real model capability
  4. colour information lost in preprocessing without being detected
  5. development data called final test or training-ready

No model, no training, no network. Reads only the recorded dev dataset.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from scarx.neural.language import tokenize  # noqa: E402

DATA = (ROOT / 'data' / 'candidates'
        / '20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646' / 'dataset.json')


@pytest.fixture(scope='module')
def data():
    return json.loads(DATA.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def token_sets(data):
    chat = set()
    for row in data['chat']:
        chat.update(tokenize(row['reference_answer']))
    vision = set()
    for row in data['vision']:
        vision.update(tokenize(row['reference_answer']))
    return {'chat': chat, 'vision': vision, 'union': chat | vision}


# ---------------------------------------------------------------- 1. 386 vs 388
def test_content_tokens_and_vocabulary_slots_are_different_quantities(token_sets):
    content = len(token_sets['union'])
    slots = content + 2  # <eos> and <bos> always occupy the first two slots
    assert content == 386, 'content token count changed; recheck the corpus'
    assert slots == 388, 'vocabulary slot count must include both control tokens'


def test_control_tokens_are_not_content_tokens(token_sets):
    """<eos>/<bos> live in the vocabulary but are never in an answer."""
    assert '<eos>' not in token_sets['union']
    assert '<bos>' not in token_sets['union']
    assert '<eos>' not in token_sets['chat']
    assert '<bos>' not in token_sets['vision']


def test_vocabulary_requires_control_tokens_in_first_two_positions():
    """The live contract the counts must satisfy."""
    import inspect
    from scarx.neural import unified
    source = inspect.getsource(unified)
    assert "['<eos>', '<bos>']" in source.replace(' ', '') or \
           "'<eos>','<bos>'" in source.replace(' ', '')
    assert '256' in source, 'vocabulary cap must still be 256'


# ------------------------------------------------ 2. intersection not subtraction
def test_union_is_not_the_naive_sum(token_sets):
    naive = len(token_sets['chat']) + len(token_sets['vision'])
    overlap = len(token_sets['chat'] & token_sets['vision'])
    assert naive != len(token_sets['union']), 'the two sets do overlap; sum is wrong'
    assert naive - overlap == len(token_sets['union'])


def test_unknown_token_count_must_use_intersection_not_size_difference(token_sets):
    """Subtracting vocabulary SIZE cannot tell you which tokens are missing.

    Demonstrate why: two disjoint vocabularies of equal size give a size
    difference of 0 while sharing nothing at all.
    """
    needed = set('abcd')
    live_equal_size_disjoint = set('wxyz')
    assert len(needed) - len(live_equal_size_disjoint) == 0
    assert len(needed & live_equal_size_disjoint) == 0
    # The size subtraction reports "nothing missing" while nothing is present.
    assert (len(needed) - len(live_equal_size_disjoint)) != \
           len(needed - live_equal_size_disjoint)


# ------------------------------------------- 3. question-only is not capability
def test_question_only_baseline_is_reported_as_baseline(data):
    by_prompt = defaultdict(Counter)
    for row in data['vision']:
        by_prompt[row['input']['prompt']][row['semantic_target']] += 1
    total = len(data['vision'])
    blind = sum(max(c.values()) for c in by_prompt.values())
    assert blind == 224
    assert abs(blind / total - 0.38888888) < 1e-6, 'documented as 38.89%'


def test_chance_baseline_split_by_class_count(data):
    by_prompt = defaultdict(Counter)
    for row in data['vision']:
        by_prompt[row['input']['prompt']][row['semantic_target']] += 1
    binary = [c for c in by_prompt.values() if len(c) == 2]
    three = [c for c in by_prompt.values() if len(c) == 3]
    assert sum(sum(c.values()) for c in binary) == 192
    assert sum(sum(c.values()) for c in three) == 384
    # Balanced within each question -> majority equals uniform chance.
    for pool, k in ((binary, 2), (three, 3)):
        for c in pool:
            n = sum(c.values())
            assert abs(max(c.values()) / n - 1 / k) < 1e-9


# --------------------------------------------------- 4. colour loss is detected
def test_grayscale_step_merges_colour_pairs(data):
    """The pipeline must be shown to lose colour, and the test must catch it."""
    import io
    import numpy as np
    from PIL import Image

    imgdir = DATA.parent / 'images'
    by_group = defaultdict(set)
    for scene in data['scenes']:
        by_group[scene['counterfactual_group']].add(scene['file'])

    merged = 0
    pairs = 0
    for files in by_group.values():
        if len(files) != 2:
            continue
        keys = []
        for rel in files:
            raw = (imgdir.parent / rel).read_bytes() if not (imgdir.parent / rel).is_file() \
                else (imgdir.parent / rel).read_bytes()
            with Image.open(io.BytesIO(raw)) as img:
                keys.append(np.asarray(img.convert('L').resize((8, 8)), np.uint8).tobytes())
        pairs += 1
        if keys[0] == keys[1]:
            merged += 1
    assert pairs == 18, 'expected 18 pure colour-swap pairs'
    assert merged == pairs, 'grayscale conversion should merge every colour pair'


# --------------------------------------- 5. development data is not final test
def test_dataset_is_development_only_and_not_training_ready(data):
    assert data.get('all_development') is True
    assert data.get('new_training') is False
    assert data.get('automatic_learning_allowed') is False
    assert data['format'] == 'scarx-grounded-development-v1'


def test_every_row_forbids_automatic_learning(data):
    assert all(r['automatic_learning_allowed'] is False for r in data['chat'])
    assert all(r['automatic_learning_allowed'] is False for r in data['vision'])


def test_grounding_decision_is_not_training_ready():
    art = json.loads(
        (ROOT / 'artifacts' / 'adaptive-curriculum-grounding.json').read_text(encoding='utf-8'))
    assert art['decision'] == 'DATA_GROUNDED_NOT_TRAINING_READY'
    assert art['automatic_learning_allowed'] is False
    assert art['live_weights_unchanged_during_build'] is True
