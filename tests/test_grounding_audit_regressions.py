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


def test_vocabulary_shortfall_is_measured_in_slots_not_content_tokens(token_sets):
    """132 missing SLOTS, not 130 content tokens.

    The artifact's needed_vocabulary_size (388) includes <eos>/<bos>, so the
    shortfall against the 256 cap is 388-256 = 132 slots. Reporting 130 (the
    content-only difference) understates what the vocabulary must allocate.
    """
    content = len(token_sets['union'])
    slots_needed = content + 2
    cap = 256
    assert content == 386
    assert slots_needed == 388
    assert slots_needed - cap == 132, 'slot shortfall must be 132'
    assert content - cap == 130, 'content-only overflow is 130 and is a different number'
    assert slots_needed - cap != content - cap
    # The artifact must agree with the slot basis.
    art = json.loads(
        (ROOT / 'artifacts' / 'adaptive-curriculum-grounding.json').read_text(encoding='utf-8'))
    assert art['compatibility']['combined_public_release']['needed_vocabulary_size'] == 388


RELEASE_CHECKPOINT = (ROOT / 'artifacts' / 'generation1'
                      / '20261109T000625Z-generation1-joint-bd6c92'
                      / 'seed-17.checkpoint.zip')


def _release_vocabulary():
    """Vocabulary of the COMMITTED generation1 release checkpoint.

    Uses the versioned artifact, never the private live session. The live session
    is not committed and does not exist on a clean clone, so a test that read it
    could not run there. This checkpoint is in git and carries the same 24-token
    vocabulary the product ships with.
    """
    from scarx.neural.unified import UnifiedCore
    return set(UnifiedCore.loads(RELEASE_CHECKPOINT.read_bytes()).vocabulary)


def test_fully_known_reference_count_is_one_and_that_is_not_a_contradiction(data):
    """Artifact says fully_known_reference_count = 1; docs used to say 'all'.

    Verified against the COMMITTED release checkpoint, so this runs on a clean
    clone with no local/ session. The vocabulary itself is never printed.
    """
    live = _release_vocabulary()

    chat_fully = [r['id'] for r in data['chat']
                  if not (set(tokenize(r['reference_answer'])) - live)]
    vis_fully = [r['id'] for r in data['vision']
                 if not (set(tokenize(r['reference_answer'])) - live)]

    art = json.loads(
        (ROOT / 'artifacts' / 'adaptive-curriculum-grounding.json').read_text(encoding='utf-8'))
    assert art['compatibility']['chat_public_release']['fully_known_reference_count'] == 1
    assert art['compatibility']['vision_public_release']['fully_known_reference_count'] == 0

    assert len(chat_fully) == 1, f'expected exactly 1 fully-known chat reference, got {chat_fully}'
    assert chat_fully == ['chat_identity_vi_07']
    assert len(vis_fully) == 0
    # 670 of 671 references are still missing at least one token.
    assert (len(data['chat']) - len(chat_fully)) + (len(data['vision']) - len(vis_fully)) == 670


def test_release_checkpoint_vocabulary_matches_the_shipped_contract():
    """The committed checkpoint must satisfy the vocabulary contract the code enforces."""
    vocab = _release_vocabulary()
    assert len(vocab) == 24, f'release vocabulary size changed: {len(vocab)}'
    from scarx.neural.unified import UnifiedCore
    model = UnifiedCore.loads(RELEASE_CHECKPOINT.read_bytes())
    assert model.vocabulary[:2] == ['<eos>', '<bos>']
    assert 3 <= len(model.vocabulary) <= 256, 'must stay inside the code cap'


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
def test_colour_pairs_collapse_in_the_REAL_preprocessing_pipeline(data):
    """Must exercise PilotLab.pixels() itself, not a private grayscale imitation.

    The first version of this test re-implemented `img.convert('L').resize((8,8))`.
    That could pass even if the production pipeline changed to keep colour, because
    it tested my copy rather than the code the product runs. This calls the real
    entry point and compares the vectors it returns.
    """
    import base64

    from scarx.app.pilots import PilotLab

    imgdir = DATA.parent / 'images'
    by_group = defaultdict(set)
    for scene in data['scenes']:
        by_group[scene['counterfactual_group']].add(scene['file'])

    def vector(rel_path):
        raw = (DATA.parent / rel_path).read_bytes()
        uri = 'data:image/png;base64,' + base64.b64encode(raw).decode()
        pixels, _digest = PilotLab.pixels(uri)   # the REAL pipeline
        return pixels.tobytes()

    pairs = 0
    merged = 0
    distinct_vectors = set()
    for files in by_group.values():
        if len(files) != 2:
            continue
        a, b = sorted(files)
        pairs += 1
        va, vb = vector(a), vector(b)
        distinct_vectors.update({va, vb})
        if va == vb:
            merged += 1

    assert pairs == 18, 'expected 18 pure colour-swap pairs'
    assert merged == pairs, (
        'the real PilotLab.pixels pipeline should merge every colour pair; '
        'if this fails the pipeline changed and colour may now be preserved')


def test_real_pipeline_yields_64_numbers_per_image(data):
    """The documented input width, read from the production function."""
    import base64

    from scarx.app.pilots import PilotLab

    scene = data['scenes'][0]
    raw = (DATA.parent / scene['file']).read_bytes()
    uri = 'data:image/png;base64,' + base64.b64encode(raw).decode()
    pixels, _digest = PilotLab.pixels(uri)
    assert pixels.size == 64, f'PilotLab.pixels must return 64 numbers, got {pixels.size}'
    assert pixels.ndim == 2 and pixels.shape == (8, 8), 'expected the 8x8 digit grid'


def test_distinct_64px_inputs_are_fewer_than_images(data):
    """36 owned scenes must collapse to 18 distinct model inputs."""
    import base64

    from scarx.app.pilots import PilotLab

    imgdir = DATA.parent / 'images'
    vectors = set()
    count = 0
    for png in sorted(imgdir.glob('*.png')):
        uri = 'data:image/png;base64,' + base64.b64encode(png.read_bytes()).decode()
        pixels, _digest = PilotLab.pixels(uri)
        vectors.add(pixels.tobytes())
        count += 1
    assert count == 36
    assert len(vectors) == 18, (
        f'colour information loss changed: {len(vectors)} distinct inputs for {count} images')


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
