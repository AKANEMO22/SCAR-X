// Execute the shipped localhost script against a tiny DOM; no inference mock.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const assert = require("node:assert/strict");

function setup() {
  const nodes = new Map();
  const node = () => ({
    children: [], textContent: "", className: "", value: "",
    classList: { remove() {}, toggle() {} },
    append(...rows) { this.children.push(...rows); },
    prepend(row) { this.children.unshift(row); },
    replaceChildren() { this.children = []; },
    remove() {},
    insertAdjacentHTML() {},
  });
  const sandbox = {
    location: { search: "?after=7357&tab=trace" },
    URLSearchParams, Date, Number, JSON, String, Error,
    setInterval() {}, setTimeout() {},
    window: { addEventListener() {} },
    fetch: () => new Promise(() => {}), // no response; tests call the actual event renderer
    document: {
      addEventListener() {},
      getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, node());
        return nodes.get(id);
      },
      createElement: () => node(),
      querySelectorAll: () => [],
    },
  };
  vm.createContext(sandbox);
  const html = fs.readFileSync(path.join(__dirname, "../src/scarx/hybrid/web.html"), "utf8");
  vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1], sandbox);
  return { nodes, sandbox,
    event(value) {
      sandbox.fixtureEvent = value;
      vm.runInContext("labEvent(fixtureEvent)", sandbox);
    },
  };
}

test("deep link begins at requested event without changing live model counters", () => {
  const { nodes, sandbox, event } = setup();
  assert.equal(vm.runInContext("cursor", sandbox), 7357);
  event({
    actor: "orchestrator", kind: "shared_endurance.heartbeat", created: Date.now() / 1000,
    payload: { run_id: "owned-lab", elapsed_seconds: 180, video_cycles: 1, checkpoints: 1,
      neural: { device: "gpu", updates: 3, replay_items: 1 } },
  });
  const text = nodes.get("lab-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /owned-lab/);
  assert.match(text, /3 cập nhật/);
  assert.match(text, /không cộng vào bộ đếm model live/);
  assert.equal(nodes.has("updates"), false);
});

test("selected ASR runtime receipt separates actual private execution from live video and learning claims", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "ASR_profile.live_activation_verified", payload: {
    ASR_model: { name: "openai/whisper-small", precision: "originalFP32" },
    verdict: "LIVE_ASR_SOURCE_STORAGE_PERSISTENCE_DATA_VERIFIED_NOT_PRODUCTION_PASS", updates: 5,
  }});
  const text = nodes.get("asr-profile-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /Runtime đã nạp.*whisper-small/);
  assert.match(text, /Chưa có lượt phiên âm video live/);
  assert.match(text, /chứng minh tự học kỹ năng/);
  assert.equal(nodes.has("updates"), false);
  event({ actor: "orchestrator", kind: "ASR_profile.actual_selected_call", payload: {
    model: "small", language: "en", sample_id: "owned-fixture", actual_transcript: "Observed output", seconds: 2,
  }});
  const answer = nodes.get("asr-answers").children[0].children.map(n => n.textContent).join("\n");
  assert.match(answer, /Phiên âm thực: Observed output/);
  assert.match(answer, /không phải kết quả model trả lời hoặc nhãn tự học/);
});

test("input boundary audit is shown as data loss, never a learned quality gain", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "input_budget_audit.completed", payload: {
    run_id: "owned-input-fixture", failures: [],
    relevance: [{ query_words: 16, identical_inputs: false }, { query_words: 125, identical_inputs: true }],
    FTS: [{ keyword_word_position: 16, blue_source_returned: true }, { keyword_word_position: 17, blue_source_returned: false }],
    historical_predicate_scope: { max_words: 11 },
  }});
  const text = nodes.get("input-budget-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /input giống byte.*125 từ/);
  assert.match(text, /FTS bỏ sót từ khóa tại vị trí 17/);
  assert.match(text, /không giải thích failure ngắn/);
  assert.match(text, /0 neuralforward\/update\/VLMcall/);
  assert.match(text, /Không phải accuracy/);
  assert.equal(nodes.has("updates"), false);
});

test("successful reconnect clears network warning without erasing action or runtime errors", async () => {
  const { nodes, sandbox } = setup();
  vm.runInContext("error('Owned action failure: draft must stay')", sandbox);
  sandbox.fetch = async () => { throw Error("Controlled transport disconnect, no model inference"); };
  await vm.runInContext("poll()", sandbox);
  assert.match(nodes.get("connection-error").textContent, /tự thử lại/);
  assert.match(nodes.get("error").textContent, /Owned action failure/);
  const status = { phase: "fixture transport only", failure: "Owned runtime failure",
    resources: { rss_bytes: 0, limit_bytes: null }, neural: { active: 0, hidden: 128, updates: 5, hidden_updates: 5 },
    goals: [], events: [], streams: [], public_topics: [], question: null, ready: false,
  };
  sandbox.fetch = async () => ({ ok: true, json: async () => status });
  await vm.runInContext("poll()", sandbox);
  assert.equal(nodes.get("connection-error").textContent, "");
  assert.match(nodes.get("error").textContent, /Owned action failure/);
  assert.equal(nodes.get("runtime-error").textContent, "Owned runtime failure");
  status.failure = null;
  await vm.runInContext("poll()", sandbox);
  assert.equal(nodes.get("runtime-error").textContent, "");
  assert.match(nodes.get("error").textContent, /Owned action failure/);
});

test("shadow query repair reports actual SQLite cost separately from neural or live capability", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "full_query_retrieval.data_verified", payload: {
    run_id: "owned-shadow-SQL", fixture_records: 1028, ordered_short_query_compatibility_cases: 10,
    maxima: { candidate_seconds: .016, candidate_peak_python_bytes: 358157 },
  }});
  const text = nodes.get("retrieval-p0-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /1028 bản ghi/);
  assert.match(text, /10 ca ngắn giữ đúng thứ tự nguồn/);
  assert.match(text, /Chưa nạp vào live/);
  assert.match(text, /0 neural\/VLMcalls/);
  assert.equal(nodes.has("updates"), false);
});

test("stale heartbeat warns without claiming a missing process is terminal", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "shared_endurance.heartbeat",
    created: Date.now() / 1000 - 120, payload: { run_id: "stale-lab" } });
  const text = nodes.get("lab-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /chưa xác nhận tiến trình đã dừng/);
  assert.doesNotMatch(text, /Đã kết thúc/);
});

test("actual lab answer is displayed with unscored status and bounded history", () => {
  const { nodes, event } = setup();
  const row = { actor: "orchestrator", kind: "shared_endurance.measured_video",
    payload: { run_id: "lab", actual_answer: { question: "Fixture question", answer: "Observed answer" } } };
  for (let i = 0; i < 14; i++) {
    // This tiny DOM needs actual lastChild to implement the shipped cap.
    const feed = nodes.get("lab-answers");
    if (feed) Object.defineProperty(feed, "lastChild", {
      configurable: true, get() { return { remove: () => feed.children.pop() }; },
    });
    event(row);
  }
  assert.equal(nodes.get("lab-answers").children.length, 12);
  const text = nodes.get("lab-answers").children[0].children.map(n => n.textContent).join("\n");
  assert.match(text, /Observed answer/);
  assert.match(text, /chưa được chấm chất lượng/);
});

test("only orchestrator receipt updates the lab panel", () => {
  const { nodes, event } = setup();
  event({ actor: "vlm", kind: "shared_endurance.completed",
    payload: { run_id: "fabricated", verdict: "PASS" } });
  assert.equal(nodes.has("lab-summary"), false);
  event({ actor: "orchestrator", kind: "shared_endurance.completed",
    created: Date.now() / 1000, payload: { run_id: "real", verdict: "INTERRUPTED_NOT_8H_PASS" } });
  assert.match(nodes.get("lab-summary").children.map(n => n.textContent).join("\n"),
    /Đã kết thúc: INTERRUPTED_NOT_8H_PASS/);
});

test("ASR candidate gain is separate from actual downstream question failures", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "ASR_candidate.benchmark_completed",
    payload: { run_id: "ASR-dev", by_language: {
      vi: { base_micro_WER: .46, small_micro_WER: .21, utterance_pairs: 16 } } } });
  const text = nodes.get("asr-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /46.00% → 21.00%/);
  assert.match(text, /chưa thay ASR live/);
  event({ actor: "orchestrator", kind: "ASR_candidate.integrated_video_answer",
    payload: { run_id: "video-p0", ASR_model: "small", question: "Fixture rank?",
      expected: "9", actual: { answer: "15" }, strict_contract_passed: false } });
  const answer = nodes.get("asr-answers").children[0].children.map(n => n.textContent).join("\n");
  assert.match(answer, /trả lời thật: 15/);
  assert.match(answer, /Đáp án đã khóa: 9/);
  assert.match(answer, /Chưa qua/);
  assert.equal(nodes.has("updates"), false);
});

test("recovery branch panel reports a failed measured action without claiming learning", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "recovery_outcome.actual_branch",
    payload: { run_id: "owned-recovery", question: "Owned task",
      selected_action: "show_schema", actual_final: { message: "Reported completion" },
      verification: { verified_resolution: false } } });
  const text = nodes.get("recovery-summary").children.map(n => n.textContent).join("\n");
  assert.match(text, /Xem schema/);
  assert.match(text, /Reported completion/);
  assert.match(text, /Chưa qua kiểm effect/);
  assert.match(text, /chưa huấn luyện SNN/);
  assert.equal(nodes.has("updates"), false);
});

test("actual predicate training stage displays loss and heldout score separately from live", () => {
  const { nodes, event } = setup();
  event({ actor: "orchestrator", kind: "predicate_retention.actual_learning_stage",
    payload: { run_id: "owned-query-screen", arm: "protected_hidden", seed: 260113,
      stage: 4, accuracy: .55, mean_train_loss: .8, updates: 160, hidden_updates: 160 } });
  const text = nodes.get("learning-screen").children.map(n => n.textContent).join("\n");
  assert.match(text, /chặng 5\/20/);
  assert.match(text, /55.0%/);
  assert.match(text, /loss học 0.800/);
  assert.match(text, /không thuộc model live/);
  assert.equal(nodes.has("updates"), false);
});

test("counterfactual query diagnostic never claims new learning", () => {
  const { nodes, event } = setup();
  event({actor: "orchestrator", kind: "predicate_binding.actual_contrast", payload: {
    run_id: "binding-audit", seed: 260113, arm: "current_hidden", api: "FindAlarms",
    contrast: "time", both_correct: .2, pairs: 100}});
  const text = nodes.get("binding-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text, /20.0% trên 100 cặp/);
  assert.match(text, /không học mới/);
  assert.equal(nodes.has("updates"), false);
});

test("common-head diagnostic separates seen fit from heldout transfer", () => {
  const { nodes, event } = setup();
  event({actor:"orchestrator",kind:"hidden_common_probe.actual_score",payload:{
    run_id:"representation-audit",arm:"current_hidden",seed:260113,reset_hidden:false,
    train:{accuracy:.9},final_DEV:{accuracy:.52}}});
  const text = nodes.get("representation-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text,/90.0% · final DEV: 52.0%/);
  assert.match(text,/hidden bị khóa/);
  assert.equal(nodes.has("updates"), false);
  event({actor:"orchestrator",kind:"hidden_common_probe.completed",payload:{
    run_id:"representation-audit",verdict:"DIAGNOSTIC_ONLY",current_hidden_minus_frozen_common_probe:{
      mean_pp:2.333,paired_seed_family_CI95_pp:[-3.083,8.167]}}});
  const completed = nodes.get("representation-summary").children.map(n=>n.textContent).join("\n");
  assert.match(completed,/2.33 điểm % · CI95% \[-3.08; 8.17\]/);
  assert.match(completed,/Readout offline chẩn đoán/);
});

test("actual file-contract repair does not imply neural learning or all-green upstream tests", () => {
  const {nodes,event}=setup();
  event({actor:"orchestrator",kind:"file_contract.repair_verified",payload:{
    run_id:"owned-file-repair",actual_dispatcher_calls:20,target_tests_passed:11,existing_upstream_failures:9}});
  const text=nodes.get("read-contract-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text,/20 lượt công cụ thật/);
  assert.match(text,/9 test cũ Hermes chưa qua/);
  assert.match(text,/chưa là SNN tự hiểu nguyên nhân/);
  assert.equal(nodes.has("updates"),false);
  event({actor:"orchestrator",kind:"file_contract.read_paths_v2_verified",payload:{
    run_id:"owned-read-paths",actual_dispatcher_calls:18,target_contracts_passed:23,existing_upstream_failures:10}});
  const upgraded=nodes.get("read-contract-summary").children.map(n=>n.textContent).join("\n");
  assert.match(upgraded,/18 lượt công cụ thật · 23 test/);
  assert.match(upgraded,/không học hoặc sửa model neural/);
});

test("credit alignment diagnostic is not displayed as training accuracy",()=>{
  const {nodes,event}=setup();
  event({actor:"orchestrator",kind:"credit_direction.completed",payload:{
    run_id:"owned-credit-check",verdict:"DIAGNOSTIC_ONLY",groups:[{state:"final640",
      local_vs_surrogate:{cosine_median:.7242},prospective_delta_vs_surrogate:{cosine_median:.1393}}]}});
  const text=nodes.get("credit-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text,/local 0.724 · delta sau history\/energy 0.139/);
  assert.match(text,/0 bước optimizer/);
  assert.match(text,/không phải accuracy/);
  assert.equal(nodes.has("updates"),false);
});

test("cold video persistence does not turn a wrong answer into neural-retention success",()=>{
  const {nodes,event}=setup();
  event({actor:"orchestrator",kind:"video_cold_memory.actual_restart_comparison",payload:{
    run_id:"owned-cold-compare",cell:"small-SNN_frozen_existing",before_score:1,after_score:1,
    persistent_memory_and_learned_state_exact:true}});
  const text=nodes.get("cold-memory-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text,/1\/3 trước · 1\/3 sau/);
  assert.match(text,/VLM dùng chung giữ nóng/);
  assert.match(text,/Persistence không phải kỹ năng neural/);
  event({actor:"orchestrator",kind:"video_cold_memory.actual_answer",payload:{
    cell:"small-SNN_frozen_existing",phase:"cold_restart",question:"Owned rank?",
    actual:{answer:"15"},expected:"9",strict_original_contract_passed:false}});
  const answer=nodes.get("cold-memory-answers").children[0].children.map(n=>n.textContent).join("\n");
  assert.match(answer,/Trả lời thật: 15/);
  assert.match(answer,/Đáp án đã khóa: 9 · Chưa qua/);
  assert.equal(nodes.has("updates"),false);
});

test("endurance audit shows repeated-video scope and keeps another job separate",()=>{
  const {nodes,event}=setup();
  event({actor:"orchestrator",kind:"shared_endurance.heartbeat",payload:{run_id:"current-lab"},created:Date.now()/1000});
  const payload={parent_endurance_run:"current-lab",verdict:"PRELIMINARY_ONLY",
    video:{summed_repeated_video_seconds:1575,unique_original_source_paths:1},
    resources:{peak_reported_tree_RSS_bytes:5.9e9,peak_reported_tree_private_commit_bytes:8.6e9,current_gateway_in_last_sample:false}};
  event({actor:"orchestrator",kind:"endurance_data_audit.snapshot",payload});
  let text=nodes.get("endurance-scope").children.map(n=>n.textContent).join("\n");
  assert.match(text,/26.3 phút lặp · 1 nguồn/);
  assert.match(text,/không cộng hai số/);
  assert.match(text,/chưa có peak đầy đủ toàn hệ/);
  event({actor:"orchestrator",kind:"endurance_data_audit.snapshot",payload:{...payload,parent_endurance_run:"old-interrupted"}});
  text=nodes.get("endurance-scope").children.map(n=>n.textContent).join("\n");
  assert.match(text,/current-lab/);assert.doesNotMatch(text,/old-interrupted/);
});

test("ASR soak is a real component profile separate from deployed model and training",()=>{
  const {nodes,event}=setup();
  event({actor:"orchestrator",kind:"ASR_small_soak.heartbeat",payload:{run_id:"owned-asr-preflight",
    elapsed_seconds:15,requested_seconds:30,complete_calls:1,completed_audio_seconds:12,GPU_headroom:{free_MiB:1400}},created:Date.now()/1000-120});
  const preflight=nodes.get("asr-soak-summary").children.map(n=>n.textContent).join("\n");
  assert.match(preflight,/15\/30 giây/);
  assert.match(preflight,/cần kiểm PID/);
  event({actor:"orchestrator",kind:"ASR_small_soak.heartbeat",payload:{run_id:"owned-asr-soak",
    elapsed_seconds:120,requested_seconds:1800,complete_calls:8,completed_audio_seconds:90,
    GPU_headroom:{free_MiB:1400}}});
  const text=nodes.get("asr-soak-summary").children.map(n=>n.textContent).join("\n");
  assert.match(text,/2\/30 phút · 8 lượt phiên âm/);
  assert.match(text,/cấu hình tại thời điểm đo; profile Hermes hiện tại được ghi riêng/);
  event({actor:"orchestrator",kind:"ASR_small_soak.actual_transcript",payload:{attempt:8,source_id:"owned-public-audio",
    actual_transcript:"Actual fixture transcript for DOM test only",seconds:1.7,RTF:.17,native_CUDA_FP32_verified:true}});
  const output=nodes.get("asr-soak-text").children[0].children.map(n=>n.textContent).join("\n");
  assert.match(output,/RTF 0.170/);assert.match(output,/CUDA\/FP32 đã kiểm log/);
  event({actor:"orchestrator",kind:"ASR_small_soak.completed",payload:{run_id:"owned-asr-soak",
    verdict:"COMPONENT_ONLY",complete_calls:165,elapsed_seconds:1800.532,completed_audio_seconds:1808.6,
    RTF_p95:.37696,min_sampled_global_free_MiB:1086}});
  const final=nodes.get("asr-soak-summary").children.map(n=>n.textContent).join("\n");
  assert.match(final,/165 lượt hoàn tất · 30.01 phút/);
  assert.match(final,/RTFp95 0.377 · minVRAM trống đã lấy mẫu 1086 MiB/);
  assert.equal(nodes.has("updates"),false);
});
