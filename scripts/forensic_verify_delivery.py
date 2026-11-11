"""Audit artifact/coverage/protection verification; does not run a learner."""
from pathlib import Path
import hashlib,json,subprocess,sys,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/research/forensic-r6-20261111'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    required=['BASELINE_INVENTORY.md','FORENSIC_AUDIT.md','PROTOCOL_LOCKED.md','CLAIM_LEDGER.json',
      'RESULTS.md','R6D_SOURCE_AUDIT.md','R6D_CLAIM_RECOMPUTE.json','R6D_GATE_INTEGRATION.json',
      'R5B_R6B_REPRODUCTION.md','DEVIATIONS.md','PROTOCOL_R6E_DRAFT.md','PROTOCOL_R7_DRAFT.md',
      'REVIEW_GEMINI_8f24a5a.md','CHATGPT_WEB_REVIEW_BRIEF_VI.md']
    missing=[name for name in required if not (DOC/name).is_file()]
    source_paths=['src/scarx/neural','src/scarx/app','src/scarx/learning','artifacts/generation1-release.json']
    protected_diff=subprocess.check_output(['git','-C',str(ROOT),'diff','main','--',*source_paths],text=True)
    main_sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','main'],text=True).strip()
    xmls=[]
    r=read(ROOT/'artifacts/forensic-r6/baseline-sharded-receipt.json');xmls.append(ROOT/r['run']/'shard1.xml')
    for phase in ['baseline-tail','native']:
        r=read(ROOT/f'artifacts/forensic-r6/{phase}-receipt.json');xmls+=list((ROOT/r['run']).glob('*.xml'))
    xmls.append(ROOT/'artifacts/forensic-r6/independent-tests.xml')
    passed=set();failed=set();skipped=set()
    for p in xmls:
        for case in E.parse(p).findall('.//testcase'):
            node=case.attrib['classname'].replace('.','/')+'.py::'+case.attrib['name']
            if case.find('failure') is not None or case.find('error') is not None:failed.add(node)
            elif case.find('skipped') is not None:skipped.add(node)
            else:passed.add(node)
    collected=subprocess.check_output([sys.executable,'-m','pytest','--collect-only','-q'],cwd=ROOT,text=True)
    expected={l.strip() for l in collected.splitlines() if l.startswith('tests/') and '::' in l}
    missing_pass=sorted(expected-passed)
    result={'required_artifacts':required,'missing_artifacts':missing,'main_sha':main_sha,
      'protected_source_has_no_diff':not protected_diff,'live_state_accessed_by_auditor':False,
      'unique_tests_currently_collected':len(expected),'unique_tests_with_pass_evidence':len(passed&expected),
      'missing_pass_evidence':missing_pass,'unexpected_test_failures_in_successful_partitions':sorted(failed),
      'skips_replaced_by_separate_native_pass':sorted(skipped&passed),'evidence_xml':[p.relative_to(ROOT).as_posix() for p in xmls],
      'single_process_full_suite_under_2gb_passed':False,'single_process_failures_preserved':True,
      'peer_source_commits':{'r6d':'4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893','r6b':'c39e9db7771582d9933af402a209917328b71119','gemini_r6c':'8f24a5abde0598201e558eee559564ccde889145'},
      'draft_protocols_executed':False,'production_ready':False}
    result['audit_delivery_checks_pass']=not missing and not missing_pass and not failed and not protected_diff and main_sha=='495dab2f4bac0180e4d1cd28674391f055ccf09d'
    (DOC/'DELIVERY_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    return int(not result['audit_delivery_checks_pass'])


if __name__=='__main__':raise SystemExit(main())
