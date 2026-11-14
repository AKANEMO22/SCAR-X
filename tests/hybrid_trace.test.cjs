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
