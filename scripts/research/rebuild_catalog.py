"""Build navigable research inventory; never infers PASS from an author's title."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
PEER_NOTES={
'research/grok/scarx-neural-round1-20261111':('GROK-R1','NO_GO','CORRECTED_SOURCE_AND_CODEX_REVIEW','Slow trace đọc được trước học; corrected split71005 gain2,734đ, lower95−0,289đ<5đ; zero=frozen.','Không dùng split71004 bị thay thế để gọi confirmationPASS.'),
'research/grok/r2-dendritic-transfer':('GROK-R2','STOP','AUTHOR_RECORD','DC-ALIF transfer không ổn định, pair_binding ởchance; order bị ảnh hưởng.','Không chứng minh dendriticgate tổng quát; original controls/protocol giữ ởarchive.'),
'research/grok/r3-temporal-stress':('GROK-R3','DIAGNOSTIC_ONLY','AUTHOR_RECORD','Tách thiếu information khỏi head không đọc được; một shuffledprobe cao không đủ kết luận leakage.','Temporal/source conventions khác nhau phải giữ riêng, không chọn metric hậu nghiệm.'),
'research/grok/r4-online-readout-mechanism':('GROK-R4','NO_GO','AUTHOR_RECORD','Readout conditioning/clipping và onlinehead có thể thất bại dù ridge đọc được code.','Probe không thay direct; chưa có learner đủgate.'),
'research/grok/r5-independent-audit':('GROK-R5','AUDIT_AND_NEGATIVE_SCREEN','AUTHOR_RECORD','Kiểm lại cue/querytick và safenormalization; có correction vị tríquery trong shifteddata.','Headvariant làm regressquery; giữ nguyên argumentaudit.'),
'research/grok/r6-causal-readout-repair':('GROK-R6','STOP','CODEX_FORENSIC_REPRODUCTION','R6B caseA direct≈66,76% trong khi ridge≈98,58%; no-clipgain≈0,98đ cóCI bao0, order giảm18,75đ.','Đã có reproduction R5B/R6B trong forensic; không adopt unclippedhead.'),
'research/deepseek/scarx-neural-round1-20261111':('DEEPSEEK-R1','PIVOT','CORRECTED_AUTHOR_RECORD','Pair-binding screen không đạt; phân biệt sampler/featuredecay/headconditioning và nonlinearspikes.','Antigravity/DeepSeek có disclosure chung phiên; không tính đồng thuận đó thành independentreplication.'),
'research/deepseek/r2-readout-learnability':('DEEPSEEK-R2','SCOPED_AUTHOR_GO','AUTHOR_RECORD','Head conditioning/substrate được tách bằng screen và confirmation riêng.','GO của câu hỏi readout không là P1/P2/productionPASS.'),
'research/deepseek/r3-temporal-substrate':('DEEPSEEK-R3','CONFIRMATION_FAIL','AUTHOR_RECORD','Temporal-substrate screen qua nhưng confirmationfail, STOP.','Giữ cả kết quả sàng lọc thuận lợi và confirmation bất lợi.'),
'research/deepseek/r4-relational-representation':('DEEPSEEK-R4','CONFIRMATION_FAIL','AUTHOR_RECORD_WITH_PEER_AUDIT','Relational information có thể đọc bằng basis/readout khác; temporal-shift confirmationfail.','Sau đó phải kiểm independent values, querytiming và rawcontrols, không coi probe là skill.'),
'research/deepseek/r5-independent-relational-binding':('DEEPSEEK-R5','CONFIRMATION_FAIL','CODEX_FORENSIC_REPRODUCTION','Independent key/value benchmark; screen9/9 nhưng confirmation caseA fail.','Bốn abortedruns được đánh dấu; canonicalclaim phải đọc bản sửa R6D/forensic.'),
'research/deepseek/r6-learning-scaling':('DEEPSEEK-R6','SUPERSEDED_BY_R6D','RETRACTED_IN_PART','Scaling/databudget cho tín hiệu, nhưng selfaudit R6D phát hiện6defects.','Stress/delayedtransition claims bị rút; aborted/supersededruns vẫn giữ.'),
'research/deepseek/r6d-audit':('DEEPSEEK-R6D','PARTIAL_DATA_GAIN_NO_GO','CODEX_FORENSIC_SOURCE_AND_RAW_AUDIT','Correcteddataeffect≈23,05đ còn tín hiệu; distractor≈55,02% không đạt gate; stressclaimwithdrawn.','Đọc claims/metric/gates đã sửa trong forensic, không dùng reportR6A chưasửa.'),
'research/deepseek/r7-causal-hidden':('DEEPSEEK-R7','CONTINUAL_GATES_UNRESOLVED','AUTHOR_RECORD_WITH_RETRACTION_CHAIN','43roundreports: readoutoverwrite/replay/credit/geometry; nhiều claim bị chính tác giả sửa.','Round43 probe+3,69đ là DEVstdoutreconstruction10/15seeds, không fullrawprovenance/confirmation; perneuron chưa thắngscalar.'),
'research/antigravity/scarx-neural-round1-20261111':('AGY-MTB','PIVOT','CODEX_ROUND1_REVIEW','Multi-timescale readout giúp truy cập information nhưng chưa chứng minh hiddenlearning cần thiết.','Credit/readoutinformationaccess và hiddenplasticity là contrasts khác nhau.'),
'research/antigravity-gemini/scarx-neural-round1-20261111':('AGY-DCALIF','PIVOT_SCOPED','CODEX_ROUND1_REVIEW','Dendriticcoincidence có kết quả scopehẹp; transfer/control/hiddennecessity chưa đủ.','Không thay originalALIFcore bằng dendritevariant từ một screen.'),
'research/gemini/r6-local-associative-learning':('GEMINI-R6','ERRATUM_PARTIAL','CODEX_FORENSIC_REVIEW','Local association có code/raw; report có phần fabricated/unsupported đã retract.','Không dùng argumentauditv1 làm bằng chứng; đọc confirmation.json và erratum.'),
'research/gemini/r7-hidden-representation-learning':('GEMINI-R7','PARTIAL_NOT_P1','CODEX_FORENSIC_REVIEW','R7 representation/replay và cross-SNNsurvey đưa ra hướngMTSG/BFcascade.','Hiddenprobe/causal/retention và sensoryencoding phải được audit riêng; chưađủscope.'),
'research/gemini/r8-causal-learning-retention-20261112':('GEMINI-R8','REPAIRED_RESEARCH_ONLY','CODEX_R8_AUDIT_AND_REPAIR','Repair sensorychannelleakage/symbolicsnooping/multikeyencoding và gates; peer607ffca đượcpin.','Codexmatchedcontrols/confirmation là nghiên cứu tiếp có protocol riêng, không gộp thành mộtPASS.'),
'research/gemini/r9-massive-neural-search-20261112':('GEMINI-R9','REQUEST_CHANGES','CODEX_INDEPENDENT_SOURCE_RAW_REVIEW','512canonicalconfigs nhưng effective412models/464model+evalsignatures; stale screen cache và confirmation selection issues.','Hai finalcandidates failgate; nhiềuconfig không là nhiềuindependentseeds.'),
'research/gemini/r9b-p1-matched-control-20261112':('GEMINI-R9B','CONFIRMATION_FAIL','SOURCE_AND_SUMMARY_CHECKS','Snapshot/frozen-readout controls được sửa; vẫn chưa đạtP1/retention.','72config-seedrows chỉ3seedclusters; clipping không phải actualnormmatching.'),
'research/gemini/r9c-distractor-causal-localization-20261112':('GEMINI-R9C','AUDIT_AND_REPAIR','AUTHOR_RECORD_WITH_SOURCE_CHECK','Xóa accuracywaiver cho causal/shuffle, tìm gategradient và noise/distractor issues.','Localizingdistractor không tự giải learning/generalization/forgetting.'),
'research/gemini/r9c-repair-literature-acceleration-20261112':('GEMINI-R9D','REFERENCE_PROPOSAL_AND_DEV','AUTHOR_RECORD','Synthesis e-prop/DGN, gategradientrepair và các adaptation nhỏ.','License shorthand ở bản gốc có lỗi: xem pinnedupstreamlicenses của Codex; DGN đổi dynamics không tự chuyển guarantee sang ALIF.'),
'research/gemini/r10-replication-learning-bottleneck-20261112':('GEMINI-R10','REPLICATION_CLAIM_SCOPED','SOURCE_AND_SUMMARY_CHECKS_NOT_FULL_RAW_REPLICATION','Tác giả báo tái lập own/stopped/literature/ACR, phân tích gradient-to-learning và dampedreset.','R11 bổ sung errata/matcheddata; không tự gọi authorINDEPENDENT_REPLICATION_PASS là mainqualityPASS.'),
'research/gemini/r11-shared-learning-readout-20261112':('GEMINI-R11','SHARED_IID_SIGNAL_NOT_CONTINUAL','ARITHMETIC_AND_SOURCE_CHECKS','Joint4096SGD direct means Binding82,49/XOR86,33/Distractor75,39/Order88,41%; một sharedcheckpoint trong study.','Per-family refitprobes là các head riêng; jointIID không chứng minh streamretention. RLS là readoutcontrol chưa phải sửa hiddencredit.'),
'research/gemini/r12-replay-lite-continual-memory-20261112':('GEMINI-R12','REPLAY_GAIN_GATES_FAIL','RECALCULATED_SUMMARY_AND_SOURCE_LIMITS','Reservoir64 matched5120updates giảmmeanF order1 từ39,56đ xuống22,37đ; order2meanF13,45đ.','WorstseedworstF33,20/34,38đ vượt5đ; futuretaskcalibration và interleavedforgettingplaceholder được erratum, chưa strictPASS.'),
'research/gemini/r13-neural-engram-composition-20261112':('GEMINI-R13-SNAPSHOT','ARCHIVE_ONLY_INVALID_CONTROLS','PINNED_SOURCE_REVIEW_NOT_ACTIVE_R13_1','Legacyf4272ba dùngAargmax→patterns_ylookup→B; mộtcontrolchancehardcoded và frozencontrol không forward train_x.','Hai independentcores/SSDweightpaging ngoài primaryonecore; reloadaccuracy không chứng minh fullstatecheckpoint. R13.1 đang sửa/chạy, bị loại khỏicleanup.'),
}
SPECIAL=[
('CODEX-R8-CONFIRMATION','R8 confirmation12seed','VERIFIED_SCOPED','P1_FAIL_RESEARCH_ONLY','docs/research/R8_P1_CONFIRMATION_20261112.md','12freshseeds,204jobs408conditions;417.792predictionrows kiểm lại; chỉXOR đủtoàn bộP1gates.','Không tune bankđã mở; BindingOODregression/Distractor63,64%<65%.'),
('CODEX-ACR','Adaptive Credit Routing','VERIFIED_SCOPED','ACR_SPECIALIZED_GAIN','docs/research/ACR_DEV_RESULTS_20261112.md','345jobs375records,3DEVseeds,3.072.000predictionentries;groups_updirect85,03/98,11/59,24%;normmargin2/3 vàshuffle3/3.','ChưabroadADVANCE/jointretention; fixedheuristicstate router khôngthắngglobalstaticprofile; mấtstate/replayattempt ghiđủ.'),
('CODEX-LITERATURE','Literature-first e-prop/BrainTrace','VERIFIED_SCOPED','REFERENCE_MIXED','docs/research/LITERATURE_REFERENCE_RESULTS_20261112.md','132jobs,675.840predictionentries; epropmatchedDistractor60,03% nhưngBinding78,58% thấp hơnown85,42%.','Transformedadapters, khôngpaperaccuracyreproduction hoặc originalframeworkexecution; noP1/P2promotion.'),
('CAPACITY-H64-H128','H48→H64→H128 capacity','VERIFIED_SCOPED','CAPACITY_ONLY_SIGNAL','docs/research/CAPACITY_H64_H128_RESULTS_20261112.md','204newjobs+48reuse=252records;H128Own direct87,30/99,41/69,40%;RSS148,82MB/commit191,15MB.','Task fit checkpoints riêng; frozenrepresentation cũng tốt lên. Extraadvantagegate có XORceilingdisclosure, không đổiscore/gate.'),
('CODEX-OUTPUT-CODEC-P0','Output codec55token P0','VERIFIED_SCOPED','P0_TECHNICAL_PASS_NOT_RELEASE','docs/research/OUTPUT_CODEC_P0_HANDOFF_V1.md','24→55outputtokens trongtrần256; decimal/sign/parity/coldresume/rollback checks có bằng chứng.','Không trainingdataset/P1/P2 hoặc quyền deployHermes từP0.'),
('CODEX-R6-FORENSIC','R6D/R6B forensic audit','VERIFIED_SCOPED','NO_PRODUCT_PROMOTION','docs/research/forensic-r6-20261111/RESULTS.md','Independent source/raw/metric/seed checks, giữ auditfinding và reproduction receipts.','Nguồn authorclaim/withdrawn tách khỏi những phép thực sự re-scored/replayed.'),
('CODEX-FROZEN-CREDIT','Frozen/trained credit oracle','VERIFIED_SCOPED','PARTIALLY_SOLVED','docs/research/literature/trained-credit-oracle.md','180TRAINdiagnosticrows45frozencheckpoints,0learningupdates; cosinekhông tự chuyển thành directquality.','Cácmethod cóweightskhácnhau; khôngcausalranking và khônggradient qua changingnormalizer/sourceoptimizer.'),
]

def main():
    index=json.loads((ROOT/'docs/research/archive/source-index.json').read_text(encoding='utf-8'))
    bypath={(v['owner'],v['original_path']):v['local_path'] for v in index['files']};entries=[]
    for p in sorted((ROOT/'docs/research').glob('[0-9][0-9]_*.md')):
        text=p.read_text(encoding='utf-8');title=next((v[2:] for v in text.splitlines() if v.startswith('# ')),p.stem)
        entries.append({'id':'CODEX-'+p.name[:2],'kind':'historical_report_or_design','agent':'Codex','title':title,
           'status':'HISTORICAL_RECORDED','audit_level':'RECORDED_NOT_REEXECUTED_BY_CLEANUP','reports':[str(p.relative_to(ROOT)).replace('\\','/')],
           'result':'Xem kết quả/giới hạn đầy đủ trong report gốc; không nâng lịch sử thành PASS qua thao tác cleanup.',
           'caveat':'Report/design/review không đồng nhất với một independenttrial hoặc một qualityPASS.'})
    for ident,title,level,status,path,result,caveat in SPECIAL:
        entries.append({'id':ident,'kind':'study_or_audit','agent':'Codex','title':title,'status':status,'audit_level':level,'reports':[path],'result':result,'caveat':caveat})
    for source in index['branches']:
        ident,status,level,result,caveat=PEER_NOTES[source['branch']]
        paths=[bypath[(source['owner'],p)] for p in source['reports'] if (source['owner'],p) in bypath]
        protocols=[bypath[(source['owner'],p)] for p in source['protocols'] if (source['owner'],p) in bypath]
        metrics=[bypath[(source['owner'],p)] for p in source['metrics'] if (source['owner'],p) in bypath]
        entries.append({'id':ident,'kind':'author_campaign','agent':source['owner'],'title':ident,'status':status,'audit_level':level,
            'source_sha':source['sha'],'original_branch':source['branch'],
            'archive_tag':None if 'r13-neural' in source['branch'] else 'archive/2026-11-13/branches/'+source['branch'],
            'reports':paths,'protocols':protocols,'metric_receipts':metrics,'original_urls':source['source_urls'],'result':result,'caveat':caveat})
    # Every individual DeepSeek sub-round remains directly queryable, including
    # retractions; automatic title text is not a verdict inference.
    for p in sorted((ROOT/'docs/research/archive/deepseek/original/docs/research/collab/scarx-neural-round1-20261111/deepseek-r7').glob('report-round*.md')):
        num=p.stem.removeprefix('report-round');text=p.read_text(encoding='utf-8')
        status='AUTHOR_SUBROUND'
        if num in ('12','18','21','24','27','28','32','33','34','35'):status='RETRACTED_OR_SUPERSEDED'
        if num=='13':status='VOID'
        if num in ('14','16','22','26','29','36','39'):status='CORRECTION_OR_ERRATUM'
        if num=='43':status='RECONSTRUCTED_DEV_NOT_CONFIRMATION'
        entries.append({'id':'DEEPSEEK-R7-'+num,'kind':'author_subround','agent':'deepseek',
          'title':next((v[2:] for v in text.splitlines() if v.startswith('# ')),p.stem),'status':status,
          'audit_level':'AUTHOR_RECORD_READ_WITH_RETRACTION_CHAIN','reports':[str(p.relative_to(ROOT)).replace('\\','/')],
          'result':'Chi tiết số liệu, đối chứng và disclosure giữ trong report; đọc cùng ledgerR7.',
          'caveat':'Không suy PASS từ title. Các round dùng lại seed/data không phải independentconfirmation.'})
    entries.append({'id':'R13-1-ACTIVE','kind':'protected_active_experiment','agent':'gemini','title':'R13-family benchmark đang chạy (ảnh ban đầu R13.1)',
       'status':'IN_PROGRESS_EXCLUDED','audit_level':'NOT_EVALUATED','reports':[],
       'protected_worktree':'C:/Users/hachimi/Downloads/model agent/scar-x-gemini-r7',
       'protected_branch':'research/gemini/r13-1-spike-composition-vs-small-llm',
       'protected_ref_prefixes':['research/gemini/r13'],'observed_active_branch_at_cleanup':'research/gemini/r13-2-independent-composition-fair-llm',
       'result':'Không đọc/copy file chưacommit, không thaybranch/process/cache và không xóa trongcleanup.',
       'caveat':'Chưa có finalresult trongcatalog này; ancestorR13f4272ba khác activeR13.1.'})
    for entry in entries:
        entry['report_sha256']={p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in entry['reports']}
    output={'date':'2026-11-13','schema':'scarx-experiment-catalog-v1','report_digest_policy':'SHA256 of report text with CRLF normalized to LF; immutable source archive hashes remain exact original bytes',
       'counts_are':'Catalog entries, NOT independentseeds/trials or production approvals',
       'entries':entries}
    (ROOT/'docs/research/experiment-catalog.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'entries':len(entries),'author_campaigns':sum(v['kind']=='author_campaign' for v in entries),'active_excluded':1}))
if __name__=='__main__':main()
