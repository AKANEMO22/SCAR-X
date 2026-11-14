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
