"""Publish explicit evidence levels; never promote an agent's status by default."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/research/forensic-r6-20261111'


def main():
    supplied=json.loads((DOC/'inputs/CODEX_CLAIM_LEDGER.json').read_text())
    adjudications={
      'C01':('PASS','SOURCE_CHECKED_AND_AGGREGATE_RECALCULATED','R2 spike-sum vs fast-decay scoped result; not new hidden learning.','OLDER_ROUND_RECALCULATION.json'),
      'C02':('FAIL','SOURCE_CHECKED_AND_AGGREGATE_RECALCULATED','Universal readout-bottleneck claim is unsupported; source-dependent geometry and optimization both matter.','OLDER_ROUND_RECALCULATION.json'),
      'C03':('PASS','AGGREGATE_RECALCULATED_NOT_RUNTIME_REPLAYED','R5 pre-query vA/vB probes .9967448/.9973958 on3seeds; frozen coding, not synaptic acquisition.','OLDER_ROUND_RECALCULATION.json'),
      'C04':('PASS','SOURCE_CHECKED_AND_HISTORICALLY_REPRODUCED','Fixed bilinear cue map permits strong R5B ridge; no learned-hidden attribution.','R5B_R6B_RECOMPUTE.json'),
      'C05':('FAIL','HISTORICALLY_REPRODUCED_WITH_INDEPENDENT_PREDICTIONS','Head both_A .6676172 fails original .70 gate.','R5B_R6B_RECOMPUTE.json'),
      'C06':('FAIL','HISTORICALLY_REPRODUCED_WITH_INDEPENDENT_PREDICTIONS','NoClip gain .0098201 with negative CI lower and order regression .1875.','R5B_R6B_RECOMPUTE.json'),
      'C07':('PASS','HISTORICALLY_REPRODUCED_WITH_INDEPENDENT_PREDICTIONS','Scoped fixed-hidden readout data effect, not quantity-only causal proof or hidden skill.','R6D_CLAIM_RECOMPUTE.json'),
      'C08':('FAIL','HISTORICALLY_REPRODUCED_WITH_INDEPENDENT_PREDICTIONS','Matched updates do not make repetitions equivalent to broader sampled episodes; order/support still differ.','R6D_CLAIM_RECOMPUTE.json'),
      'C09':('INCONCLUSIVE','NOT_INDEPENDENTLY_TESTED','No defined independently validated delayed/emergent phase-transition result; sample ladder is not proof.','DATA_LINEAGE.json'),
      'C10':('INVALID_HARNESS','SOURCE_CHECKED_AND_MUTATION_TESTED','R6A raw/scaled stress mismatch invalidates that robustness attribution; not independently retrained R6A.','R6D_SOURCE_AUDIT.md'),
      'C11':('INVALID_HARNESS','HISTORICALLY_REPRODUCED_WITH_INDEPENDENT_PREDICTIONS','Recorded distractor .5502305 fails score threshold; same-noise causal pairing is also violated.','R6D_GATE_INTEGRATION.json'),
      'C12':('FAIL','SOURCE_CHECKED_AND_HISTORICALLY_REPRODUCED','Original matrix retains3null gates; supplemental controls do not imply same trained head.','R6D_GATE_INTEGRATION.json'),
      'C13':('PASS','SOURCE_AND_SAVED_ROW_CHECKS_NOT_FULL_TRAINING_REPLAY','DC-ALIF strict_binding result is task-scoped, under disclosed protocol deviation; not universal or release GO.','OLDER_CAMPAIGN_STATUS.md'),
      'C14':('FAIL','SOURCE_AND_SAVED_REPORT_CHECKS','DC-ALIF general relational improvement conflicts with reported order/pair-binding regressions; not all campaigns rerun.','FORENSIC_AUDIT.md'),
      'C15':('INCONCLUSIVE','NARROW_POSITIVE_HISTORY_PLUS_FROZEN_R6_AUDIT','Main20/DC-ALIF narrow gains exist; broad stable hidden representation not established. R6B/C/D hidden frozen.','FORENSIC_AUDIT.md'),
      'C16':('FAIL','HISTORICAL_SOURCE_RESULTS_NOT_P2_RETRAINING','Reported main/Gemini retention fails; this audit does not open a newP2pilot.','CHATGPT_WEB_REVIEW_BRIEF_VI.md'),
      'C17':('NOT_TESTED','AUTHORIZATION_AND_SOURCE_CHECKED','Main readout-usefulness only P0 approved; no new pilot authorized or run by Codex.','PROTOCOL_R7_DRAFT.md'),
      'C18':('FAIL','HISTORICAL_SOURCE_RESULTS','Normal dialogue not established by existing free-utterance pilot; do not use synthetic retrieval as substitute.','CHATGPT_WEB_REVIEW_BRIEF_VI.md'),
      'C19':('NOT_TESTED','HISTORICAL_SCOPE_CHECKED','General natural-image QA not demonstrated; digit8x8 serving and failed visual pilot are different scopes.','CHATGPT_WEB_REVIEW_BRIEF_VI.md'),
      'C20':('NOT_TESTED','NO_AUTHORIZED_CAPACITY_STUDY_IN_THIS_AUDIT','Capacity allocation is engineering; no inference that more neurons automatically improve skill.','PROTOCOL_R7_DRAFT.md'),
      'C21':('NOT_TESTED','USER_CONSTRAINT_PRESERVED','No external LLM/teacher used in audited candidate inference; policy is not a scientific PASS claim.','FORENSIC_AUDIT.md'),
      'C22':('FAIL','SOURCE_CHECKED','More data cannot recover information already removed by preprocessing; source-specific exact collisions/codec remain separate.','CHATGPT_WEB_REVIEW_BRIEF_VI.md'),
      'C23':('INCONCLUSIVE','BOUNDED_REGRESSION_WITH_RECORDED_RESOURCE_FAILURES','Regression partition can pass; one-process full suite exceeds cap in duplicated Torch controls. Not a serving release certificate.','DEVIATIONS.md')}
    claims=[]
    for old in supplied['claims']:
        verdict,level,reason,evidence=adjudications[old['id']]
        claims.append({'id':old['id'],'claim':old['claim_vi'],'untrusted_handoff_working_status':old['working_status'],
            'codex_verdict':verdict,'evidence_level':level,'evidence':evidence,'reason_and_limits':reason,
            'source_details':'BASELINE_INVENTORY.json and the linked audit artifact','independent_fresh_confirmation':False})
    result={'artifact_type':'SCARX_INDEPENDENT_CLAIM_LEDGER','claims':claims,
        'additional_gemini_r6c_claims':'REVIEW_GEMINI_8f24a5a.md and GEMINI_REPRODUCTION_AUDIT.json',
        'not_all_past_campaigns_runtime_reproduced':True,'production_ready':False,'main_merged':False}
    (DOC/'CLAIM_LEDGER.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'claims':len(claims),'all_have_explicit_verdict':True}))


if __name__=='__main__':main()
