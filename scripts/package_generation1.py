"""One current product release. Include only public artifacts; no local data."""
from pathlib import Path
import hashlib
import json
import zipfile
root=Path(__file__).resolve().parents[1];target=root/'artifacts/SCAR-X-Generation-1.zip'
allowed={'.py','.md','.json','.jsonl','.yaml','.toml','.txt','.css','.html','.js','.ts','.tsx','.cpp','.h','.ps1','.patch'}
files=[]
for base in ('src','docs','scripts','tests','configs','experiments','data','native','integrations','runs'):
    files.extend(p for p in (root/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in allowed)
files.extend(root/name for name in ('README.md','pyproject.toml','run_app.py','Start-SCARX.ps1','Stop-SCARX.ps1','AGENTS.md','project.json','capabilities.json','.gitignore','.ignore'))
release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
path=(root/release['checkpoint']).resolve();assert path.is_relative_to((root/'artifacts').resolve())
files.extend([path,root/'artifacts/generation1-release.json',root/'artifacts/generation1-engineering.json',root/'artifacts/generation1-live-check.json',root/'artifacts/generation1-ui.jpg',root/'artifacts/phase21-summary.json',root/'artifacts/pilots/vision-examples.json'])
files.extend(root/'artifacts'/name for name in ('hermes-stateful-check.json','hermes-stateful-stream-check.json','hermes-stateful-source-proof.json','hermes-stateful-desktop-check.json','hermes-stateful-ui.jpg'))
files.extend(root/'artifacts'/name for name in ('adaptive-generation-engineering.json','continuous-dialogue-acceptance.json',
 'antigravity-dataset-latest.json','antigravity-dataset-validation.json','antigravity-supplement-latest.json',
 'antigravity-supplement-validation.json','hermes-online-question.jpg','hermes-online-resolved.jpg','hermes-online-desktop-check.json'))
files.append(root/'artifacts/continuous-dialogue-delivery.json')
files.extend(root/'artifacts'/name for name in ('knowledge-retention-latest.json',
 'knowledge-retention-audit-latest.json','qwen4b-reference-download.json','qwen4b-comparison-latest.json'))
files.extend(root/'artifacts'/name for name in ('hermes-learning-channel-check.json','hermes-learning-tab-delivery.json'))
files.extend(root/'artifacts'/name for name in ('p1-input-information-audit.json','p1-input-events-latest.json',
 'p1-local-target-latest.json','p1-random-feedback-latest.json','p2-readout-conditioning-latest.json',
 'p2-conditioning-confirmation-latest.json','antigravity-events-latest.json','antigravity-events-validation.json',
 'event-dialogue-latest.json','event-dialogue-cold-check.json','qwen-dialogue-reference-latest.json'))
files.append(root/'artifacts/p1-p2-event-progress.json')
files.extend(root/'artifacts'/name for name in ('byte-dialogue-latest.json','byte-bptt-control-latest.json',
 'decoder-state-retention-audit.json','byte-working-memory-latest.json','rgb-object-dataset.json',
 'rgb-object-download-failure.json','rgb-object-grounding-latest.json','rgb-object-grounding-audit.json'))
files.append(root/'artifacts/byte-memory-rgb-progress.json')
files.extend(root/'artifacts'/name for name in ('p1-local-classifier-latest.json',
 'live-observatory-latest.json','passive-trace-delivery.json',
 'capacity-live-observation.json','neuron-observatory-ui.png',
 'passive-answer-trace.png','passive-learning-trace.png'))
files.extend(root/'artifacts'/name for name in ('p1-local-classifier-v2-latest.json',
 'p1-local-classifier-v2-audit.json','p1-local-classifier-v2-delivery.json',
 'p1-v2-running.png','p1-v2-final.png'))
files.extend(root/'artifacts'/name for name in ('p1-control-factors-latest.json',
 'p1-control-factors-delivery.json','p1-control-factors-running.png',
 'p1-control-factors-final.png'))
files.extend(root/'artifacts'/name for name in ('p1-score-integrity-latest.json',
 'p1-paired-score-v2-latest.json','p1-paired-score-v2-audit.json',
 'p1-paired-score-v2-delivery.json','paired-score-final.png'))
files.extend(root/'artifacts'/name for name in ('p2-paired-score-diagnostic-latest.json',
 'p2-cue-information-latest.json','p2-paired-score-diagnostic-delivery.json',
 'p2-paired-retention-running.png','p2-paired-retention-final.png'))
files.extend(root/'artifacts'/name for name in ('p2-inference-dynamics-latest.json','p2-timescale-view-latest.json',
 'antigravity-g1-state-latest.json','g1-state-prompts-reviewed.json','g1-conversational-state-latest.json',
 'g1-basic-rehearsal-latest.json','g1-surface-acceptance-latest.json','qwen-g1-state-reference-latest.json',
 'antigravity-g1-curriculum-latest.json','g1-curriculum-reviewed.json','g1-curriculum-usage.json','g1-curated-curriculum-latest.json',
 'g1-project-supervision-latest.json','g1-guard-stop-latest.json','g1-post-supervision-regression.json',
 'g1-live-new-prompts-latest.json','g1-dialogue-curriculum-delivery.json',
 'g1-live-project-rollback.png','g1-live-qwen-trace.png','g1-guard-stop-trace.png'))
files.extend(root/'artifacts'/name for name in ('g1-representation-decoder-latest.json',
 'g1-bounded-beam-latest.json','p2-neuron-state-observability-latest.json',
 'g1-word-score-transfer-latest.json','g1-input-identity-latest.json',
 'passive-work-trace-latest.json','passive-work-trace-delivery.json','passive-work-trace.png','neural-core-boundary.json'))
files.extend(root/'artifacts'/name for name in ('g1-word-score-inputs-latest.json','g1-word-score-inputs-audit.json',
 'g1-word-input-legacy-parity.json','g1-word-input-data-usage.json','g1-word-conditioning-latest.json',
 'g1-word-input-delivery.json','g1-word-input-running.png','g1-word-input-final.png'))
files.extend(root/'artifacts'/name for name in ('g1-word-control-preflight.json','g1-word-learnability-control-latest.json',
 'g1-word-control-io-failure.json','g1-observer-atomic-recovery.json','qwen-word-control-reference-latest.json',
 'g1-word-control-data-usage.json','g1-word-control-delivery.json','g1-word-control-running.png',
 'g1-word-control-all-cases.png','g1-word-control-final.png'))
files.extend(root/'artifacts'/name for name in ('g1-credit-noise-latest.json','g1-credit-noise-delivery.json',
 'g1-credit-noise-final.png'))
files.extend(root/'artifacts'/name for name in ('g1-observed-association-latest.json','g1-observed-association-audit.json',
 'g1-observed-association-data-usage.json','g1-observed-association-delivery.json',
 'g1-observed-association-final.png'))
files.extend(root/'artifacts'/name for name in ('antigravity-vision-questions-latest.json','vision-questions-reviewed.json',
 'vision-question-path-latest.json','vision-question-path-delivery.json','vision-question-path-final.png'))
files.extend(root/'artifacts'/name for name in ('visual-word-dataset-latest.json','visual-word-readout-latest.json',
 'visual-word-readout-audit.json','visual-word-data-usage.json','visual-word-readout-delivery.json','visual-word-readout-final.png'))
files.extend(root/'artifacts'/name for name in ('visual-word-control-preflight.json','visual-word-control-latest.json',
 'visual-word-control-data-usage.json','visual-word-control-delivery.json','visual-word-control-running.png',
 'visual-word-control-results.png','visual-word-control-final.png'))
files.extend(root/'artifacts'/name for name in ('visual-information-latest.json','visual-information-data-usage.json',
 'visual-information-delivery.json','visual-information-final.png'))
files.extend(root/'artifacts'/name for name in ('direct-eligibility-engineering.json','direct-eligibility-latest.json',
 'direct-eligibility-read-io-failure.json','direct-eligibility-data-usage.json','direct-eligibility-delivery.json',
 'direct-eligibility-final.png','direct-eligibility-answers.png'))
files.extend(root/'artifacts'/name for name in ('antigravity-direct-word-prompts.json','antigravity-direct-word-repair.json',
 'direct-word-data-reviewed.json','direct-word-preflight.json','direct-word-bridge-latest.json','qwen-direct-word-reference.json',
 'direct-word-software-validation.json','direct-word-data-usage.json','direct-word-delivery.json',
 'direct-word-results.png','direct-word-answers.png'))
files.extend(root/'artifacts'/name for name in ('direct-visual-preflight.json','direct-visual-bridge-latest.json',
 'direct-visual-software-validation.json','direct-visual-data-usage.json','direct-visual-delivery.json',
 'direct-visual-running.png','direct-visual-results.png','direct-visual-answers.png'))
files.extend(root/'artifacts'/name for name in ('local-credit-alignment.json',
 'deepseek-product-review.json','deepseek-review-software-validation-initial.json',
 'deepseek-review-software-validation.json','deepseek-review-live-preservation.json',
 'deepseek-review-acceptance.json','deepseek-review-delivery.json','deepseek-review-final.jpg'))
files.extend(root/'artifacts'/name for name in ('adaptive-curriculum-data.json',
 'adaptive-curriculum-reviewed.json','adaptive-curriculum-reference.json',
 'adaptive-curriculum-software-validation.json','adaptive-curriculum-delivery.json',
 'adaptive-curriculum-reference-final.jpg'))
# As-executed source snapshots, never private CLI logs/session DB/weights.
for receipt_name in ('adaptive-curriculum-data.json','adaptive-curriculum-reviewed.json','adaptive-curriculum-reference.json'):
    receipt=json.loads((root/'artifacts'/receipt_name).read_text(encoding='utf-8'))
    snapshot=(root/receipt['run']/'source.zip').resolve()
    assert snapshot.is_relative_to((root/'runs').resolve()) and snapshot.is_file()
    files.append(snapshot)
files.append(root/'runs/20261110T131337Z-adaptive-curriculum-data-v1-3c0f41/source.zip')
# These12images are owned syntheticrenderer fixtures, not arbitrary userphotos.
vision_receipt=root/'artifacts/vision-question-path-latest.json'
if vision_receipt.is_file():
    vision=json.loads(vision_receipt.read_text(encoding='utf-8'));metadata=(root/vision['data']).resolve()
    assert metadata.is_relative_to((root/'data/candidates').resolve()) and metadata.name=='qa-fixtures.json'
    assert hashlib.sha256(metadata.read_bytes()).hexdigest()==vision['data_sha256']
    fixtures=json.loads(metadata.read_text(encoding='utf-8'));assert len(fixtures['scenes'])==12 and fixtures['automatic_learning_allowed'] is False
    for scene in fixtures['scenes']:
        assert scene['file'] in {f's{i:02d}.png' for i in range(1,13)}
        p=(metadata.parent/scene['file']).resolve();assert p.is_relative_to(metadata.parent) and p.stat().st_size<10000 and hashlib.sha256(p.read_bytes()).hexdigest()==scene['sha256']
        files.append(p)
# Jointwordresearch uses192ownedrenderings, never arbitrary userimages.
visual_dataset_receipt=root/'artifacts/visual-word-dataset-latest.json'
if visual_dataset_receipt.is_file():
    visual_dataset=json.loads(visual_dataset_receipt.read_text(encoding='utf-8'));metadata=(root/visual_dataset['data']).resolve()
    assert metadata.is_relative_to((root/'data/candidates').resolve()) and metadata.name=='dataset.json' and hashlib.sha256(metadata.read_bytes()).hexdigest()==visual_dataset['sha256']
    visual_data=json.loads(metadata.read_text(encoding='utf-8'));assert len(visual_data['scenes'])==192 and visual_data['automatic_learning_allowed'] is False
    allowed_visual_names={f'{split}-{i:03d}.png' for split in ('train','development') for i in range(1,97)}
    for scene in visual_data['scenes']:
        assert scene['file'] in allowed_visual_names
        p=(metadata.parent/scene['file']).resolve();assert p.is_relative_to(metadata.parent) and p.stat().st_size<10000 and hashlib.sha256(p.read_bytes()).hexdigest()==scene['sha256']
        files.append(p)
delivery=json.loads((root/'artifacts/continuous-dialogue-delivery.json').read_text(encoding='utf-8'))
files.append(root/delivery['run']/'source.zip')
# Only the two attributed public photos, never arbitrary local image files.
supplement=json.loads((root/'artifacts/antigravity-supplement-latest.json').read_text(encoding='utf-8'))
photo_dir=(root/supplement['candidate_directory']).resolve()
assert photo_dir.is_relative_to((root/'data/candidates').resolve())
files.extend(photo_dir/name for name in ('china.jpg','flower.jpg'))
# Historical public pilot fixtures support legacy regression tests offline;
# never loaded by Generation1, never listed as active product models.
historical=json.loads((root/'artifacts/pilot-release.json').read_text(encoding='utf-8'))
files.append(root/'artifacts/pilot-release.json')
for entry in historical.values():
    p=(root/entry['checkpoint']).resolve();assert p.is_relative_to((root/'artifacts').resolve())
    files.extend(p.rglob('*') if p.is_dir() else [p])
# Preserve source as executed for the new training run and its audit.
files.append(root/release['run']/'source.zip')
files=sorted({p.resolve() for p in files if p.is_file()})
for p in files:
    relative=p.relative_to(root.resolve())
    assert 'local' not in relative.parts and '.env' not in p.name and 'backups' not in relative.parts
manifest={p.relative_to(root.resolve()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for p in files:archive.writestr('scar-x/'+p.relative_to(root.resolve()).as_posix(),p.read_bytes())
    archive.writestr('scar-x/PACKAGE_MANIFEST.json',json.dumps({'product':'SCAR-X Generation1','active_model':'scarx-generation1','runtime_core_count':1,'initial_checkpoint':release['checkpoint'],'historical_pilot_artifacts':'offline regression fixtures, not active runtime models','sha256':manifest,'excludes':['local/','Hermes personal config/backups','keys','.env','external models','old release ZIPs']},indent=2))
with zipfile.ZipFile(target) as archive:
    assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==digest for name,digest in manifest.items())
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(root/'artifacts/SCAR-X-Generation-1.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii')
print(json.dumps({'package':str(target),'bytes':target.stat().st_size,'files':len(files),'sha256':digest}))
