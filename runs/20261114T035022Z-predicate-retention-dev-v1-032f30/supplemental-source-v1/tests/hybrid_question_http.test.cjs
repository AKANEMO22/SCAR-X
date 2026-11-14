// Actual shipped event handlers -> real HTTP -> isolated HybridEngine.feedback.
// This is a UI transport fixture, not a model response or learning benchmark.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const assert = require("node:assert/strict");
const file = process.env.SCARX_QUESTION_TEST_SOURCE ||
  path.join(__dirname, "../integrations/hermes/hybrid_plugin/plugin.js");
const source = fs.readFileSync(file, "utf8")
  .replace(/^import[\s\S]*?from\s+["'][^"']+["'];?\s*/gm, "")
  .replace("export default", "globalThis.scarxPlugin =");
const base = process.env.SCARX_QUESTION_TEST_BASE;
const nonce = process.env.SCARX_QUESTION_TEST_NONCE;

test("question click keeps failed draft then sends once through real isolated HTTP", { skip: !base }, async () => {
  const state = [], refs = [];
  let index = 0, refIndex = 0, query;
  async function request(endpoint, options = {}) {
    const response = await fetch(base + endpoint, {
      method: options.method || "GET",
      headers: { "Content-Type": "application/json", "X-UI-Fixture": nonce },
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    const value = await response.json();
    if (!response.ok) throw Error(value.error);
    return value;
  }
  const sandbox = {
    host: { state: { model: {} }, notifyError() {} },
    useValue: () => "scarx-hybrid",
    useState(initial) {
      const i = index++;
      if (!(i in state)) state[i] = initial;
      return [state[i], value => { state[i] = value; }];
    },
    useRef(initial) {
      const i = refIndex++;
      if (!(i in refs)) refs[i] = { current: initial };
      return refs[i];
    },
    useEffect() {},
    useQuery: () => query,
    jsx: (tag, props) => ({ tag, ...props }),
    jsxs: (tag, props) => ({ tag, ...props }),
  };
  sandbox.ctx = { rest: request };
  query = { data: await request("/status"), refetch: async () => {
    query.data = await request("/status");
  }};
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: file });
  function render() {
    index = refIndex = 0;
    return vm.runInContext("LearningQuestion({ctx})", sandbox);
  }
  const descend = n => [n, ...(Array.isArray(n?.children) ? n.children.flatMap(descend) : [])];
  const event = () => ({ preventDefault() {}, stopPropagation() {} });
  let widget = render();
  assert.notEqual(widget.tag, "form", "A plugin in Hermes composer cannot nest its own form");
  const textarea = descend(widget).find(n => n.tag === "textarea");
  textarea.onChange({ target: { value: "Owned UI transport fixture reply" } });
  widget = render();
  let send = descend(widget).find(n => n.className === "sxq-submit");
  assert.equal(send.type, "button");
  assert.equal(typeof send.onClick, "function");
  send.onClick(event());
  const until = Date.now() + 3000;
  while (Date.now() < until && !(await request("/status")).attempts) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  await new Promise(resolve => setTimeout(resolve, 40));
  widget = render();
  assert.equal(descend(widget).find(n => n.tag === "textarea").value, "Owned UI transport fixture reply");
  assert.ok(descend(widget).some(n => n.role === "alert"), "Network failure must be visible on the card");
  send = descend(widget).find(n => n.className === "sxq-submit");
  send.onClick(event());
  // Same rendered callback double-click must not issue duplicate feedback.
  send.onClick(event());
  const deadline = Date.now() + 3000;
  while (Date.now() < deadline && (await request("/status")).question) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  await new Promise(resolve => setTimeout(resolve, 40));
  widget = render();
  assert.ok(descend(widget).some(n => n.role === "status"), "Successful backend acknowledgement stays visible");
  const status = await request("/status");
  assert.equal(status.attempts, 2);
  assert.equal(status.updates, 0);
  assert.equal(status.hidden_updates, 0);
  assert.equal(status.last_feedback.text, "Owned UI transport fixture reply");
  assert.equal(Object.hasOwn(status.last_feedback, "helpful"), false);
  assert.equal(status.question, null);
});
