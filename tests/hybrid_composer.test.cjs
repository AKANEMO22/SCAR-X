// Executes the shipped disk plugin's actual middleware with the host SDK stubbed.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const assert = require("node:assert/strict");
const { webcrypto } = require("node:crypto");
const file = path.join(
  __dirname,
  "../integrations/hermes/hybrid_plugin/plugin.js",
);
const source = fs
  .readFileSync(file, "utf8")
  .replace(/^import[\s\S]*?from\s+["'][^"']+["'];?\s*/gm, "")
  .replace("export default", "globalThis.scarxPlugin =");

function setup(sessions = ["chat-a", "stored-a"]) {
  const calls = [],
    storage = new Map([
      [
        "video:chat-a",
        {
          id: "c4ac181367fc427593d7b5439f2558fd",
          label: "SCAR-X engineering capture fixture",
        },
      ],
      ["video:stored-a", { id: "old" }],
      ["video:draft", { id: "old" }],
    ]);
  const sandbox = {
    crypto: webcrypto,
    TextEncoder,
    Date,
    Map,
    Set,
    JSON,
    document: {},
    useRef: (initial) => ({ current: initial }),
    host: {
      composer: { preparedSessionVersion: 1 },
      state: {
        model: { get: () => "scarx-hybrid" },
        focusedSessionId: { get: () => sessions[0] },
        focusedStoredSessionId: { get: () => sessions[1] },
      },
      notifyError: (message) => calls.push({ error: message }),
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: file });
  sandbox.ctx = {
    storage: {
      get: () => {
        throw Error("Legacy sticky video must never be read");
      },
      remove: (key) => storage.delete(key),
    },
    rest: async (endpoint, options) => {
      calls.push({ endpoint, body: options.body });
      return endpoint === "/feedback" ? { accepted: true } : endpoint === "/observe"
        ? { id: "a".repeat(32) }
        : { queued: true };
    },
  };
  return { sandbox, calls, storage };
}

test("learning replies reward the controller only with an explicit rating", async () => {
  for (const [rating, skip] of [
    [undefined, false], [true, false], [false, false],
    [undefined, true], [true, true], [false, true],
  ]) {
    const { sandbox, calls } = setup();
    const state = ["Owner guidance", false, null, "", null];
    let cursor = 0;
    sandbox.useValue = (atom) => atom.get();
    sandbox.useState = () => {
      const index = cursor++;
      return [state[index], (value) => { state[index] = value; }];
    };
    sandbox.useQuery = () => ({
      data: { question: { id: "pending" } },
      refetch: async () => {},
    });
    sandbox.useEffect = () => {};
    sandbox.jsx = sandbox.jsxs = (tag, props) => ({ tag, ...props });
    const render = () => {
      cursor = 0;
      return vm.runInContext("LearningQuestion({ctx})", sandbox);
    };
    const descendants = (n) => [n, ...(Array.isArray(n?.children) ? n.children.flatMap(descendants) : [])];
    let widget = render();
    assert.equal(widget.tag, "div");
    assert.ok(descendants(widget).some(n => n.tag === "textarea"));
    if (typeof rating === "boolean") {
      const button = descendants(widget).find(n => n["aria-label"] === (rating ? "Hữu ích" : "Chưa hữu ích"));
      button.onClick();
      assert.equal(calls.length, 0, "Selecting a rating cannot create a learning label yet");
      widget = render();
    }
    if (skip) {
      descendants(widget).find(n => n.children === "Bỏ qua").onClick();
    } else {
      descendants(widget).find(n => n.className === "sxq-submit").onClick({
        preventDefault() {}, stopPropagation() {},
      });
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls.length, 1);
    assert.equal(calls[0].endpoint, "/feedback");
    const expectedRating = skip ? undefined : rating;
    assert.equal(calls[0].body.helpful, expectedRating);
    assert.equal(Object.hasOwn(calls[0].body, "helpful"), typeof expectedRating === "boolean");
  }
});

test("negative rating can be sent without inventing a typed reply; blank default cannot submit", async () => {
  for (const rating of [null, false]) {
    const { sandbox, calls } = setup();
    let cursor = 0;
    const state = ["", false, rating, "", null];
    sandbox.useValue = atom => atom.get();
    sandbox.useState = () => [state[cursor++], () => {}];
    sandbox.useQuery = () => ({ data: { question: { id: "pending", question: "Question" } },
      refetch: async () => {} });
    sandbox.useEffect = () => {};
    sandbox.jsx = sandbox.jsxs = (tag, props) => ({ tag, ...props });
    const widget = vm.runInContext("LearningQuestion({ctx})", sandbox);
    const all = (n) => [n, ...(Array.isArray(n?.children) ? n.children.flatMap(all) : [])];
    all(widget).find(n => n.className === "sxq-submit").onClick({
      preventDefault() {}, stopPropagation() {},
    });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls.length, rating === false ? 1 : 0);
    if (calls.length) assert.equal(calls[0].body.helpful, false);
  }
});

test("debug learning tab uses the same guarded reply component as the chat card", () => {
  const { sandbox } = setup();
  let cursor = 0;
  const states = ["learn", "", "", "", "", "", false, [], [], "", null];
  sandbox.useState = () => [states[cursor++], () => {}];
  sandbox.useQuery = () => ({ data: { question: { id: "debug-pending", question: "Owned fixture" } } });
  sandbox.useEffect = () => {};
  sandbox.jsx = sandbox.jsxs = (tag, props) => ({ tag, ...props });
  const panel = vm.runInContext("Panel({ctx})", sandbox);
  const all = n => [n, ...(Array.isArray(n?.children) ? n.children.flatMap(all) : [])];
  const shared = all(panel).filter(n => typeof n?.tag === "function" && n.tag.name === "LearningQuestion");
  assert.equal(shared.length, 1);
  assert.equal(shared[0].debug, true);
});

test("plain turn stays identical despite stale engineering fixture storage", async () => {
  const { sandbox, calls, storage } = setup();
  sandbox.draft = { text: "câu này ntn", attachments: [] };
  const result = await vm.runInContext("attachVideos(ctx,draft)", sandbox);
  assert.strictEqual(result, sandbox.draft);
  assert.equal(result.text, "câu này ntn");
  assert.equal(calls.length, 0);
  assert.equal(storage.size, 0);
});
test("native video uses hidden metadata without changing typed text", async () => {
  const { sandbox, calls } = setup();
  sandbox.draft = {
    text: "  Giải thích đoạn này?\n",
    attachments: [
      { id: "file-v", path: "C:\\Videos\\my video.mp4", label: "my video.mp4" },
      { id: "image", path: "C:\\image.png" },
    ],
  };
  const before = JSON.stringify(sandbox.draft);
  const result = await vm.runInContext("attachVideos(ctx,draft)", sandbox);
  assert.equal(result.text, sandbox.draft.text);
  assert.equal(JSON.stringify(sandbox.draft), before);
  assert.equal(result.attachments.length, 1);
  assert.equal(result.attachments[0].id, "image");
  assert.equal(calls.length,0,'No unscoped upload is queued before the host resolves the admission');
  await result.prepareSession({runtimeSessionId:'chat-a',storedSessionId:'stored-a',submissionId:'exact-input',text:sandbox.draft.text});
  assert.deepEqual(
    calls.map((c) => c.endpoint),
    ["/observe", "/video-ready", "/observe"],
  );
  assert.equal(calls[2].body.existing_stream_id, "a".repeat(32));
  assert.match(calls[2].body.video_context.text_sha256, /^[a-f0-9]{64}$/);
  assert.equal(calls[2].body.video_context.submission_id,'exact-input');
  assert.equal(
    JSON.stringify(calls[2].body.video_context.session_ids),
    JSON.stringify(["chat-a", "stored-a"]),
  );
  assert.ok(!result.text.includes("scarx://"));
  assert.ok(!result.text.includes("[Video:"));
});
test("fresh draft has no persistent global draft context", async () => {
  const { sandbox, calls } = setup([null, null]);
  sandbox.draft = {
    text: "xem?",
    attachments: [{ path: "C:\\video.webm", id: "v" }],
  };
  const result = await vm.runInContext("attachVideos(ctx,draft)", sandbox);
  assert.equal(result.text, "xem?");
  assert.equal(calls.length,0);
  await result.prepareSession({runtimeSessionId:'created-for-this-draft',storedSessionId:null,submissionId:'first-turn',text:'xem?'});
  assert.deepEqual([...calls[2].body.video_context.session_ids],['created-for-this-draft']);
  assert.equal(calls[2].body.video_context.submission_id,'first-turn');
});
test("upload failure preserves original draft and cancels sending", async () => {
  const { sandbox, calls } = setup();
  sandbox.ctx.rest = async () => {
    throw Error("fixture upload failure");
  };
  sandbox.draft = {
    text: "không được mất chữ",
    attachments: [{ path: "C:\\video.mp4", id: "v" }],
  };
  const before = JSON.stringify(sandbox.draft);
  const prepared=await vm.runInContext("attachVideos(ctx,draft)", sandbox);
  await assert.rejects(prepared.prepareSession({runtimeSessionId:'chat-a',storedSessionId:null,submissionId:'owned',text:'không được mất chữ'}),/fixture upload failure/);
  assert.equal(JSON.stringify(sandbox.draft), before);
  assert.equal(calls.length,0,'Host rejection restores the original draft; no video binding was committed');
});

test('same text in two first drafts is owned by two actual submissions, not global focus',async()=>{
  const {sandbox,calls}=setup([null,null]);
  sandbox.draft={text:'xem giúp',attachments:[{id:'video',path:'C:\\owned.mp4'}]};
  const a=await vm.runInContext('attachVideos(ctx,draft)',sandbox);
  const b=await vm.runInContext('attachVideos(ctx,draft)',sandbox);
  await a.prepareSession({runtimeSessionId:'session-A',storedSessionId:null,submissionId:'input-A',text:'xem giúp'});
  await b.prepareSession({runtimeSessionId:'session-B',storedSessionId:null,submissionId:'input-B',text:'xem giúp'});
  const bindings=calls.filter(c=>c.body.video_context).map(c=>c.body.video_context);
  assert.deepEqual(bindings.map(b=>[b.session_ids[0],b.submission_id]),[['session-A','input-A'],['session-B','input-B']]);
  assert.notEqual(bindings[0].ticket_id,bindings[1].ticket_id);
});

test('retry after renderer reload derives the same upload operation id',async()=>{
  const ids=[];
  for(let i=0;i<2;i++){
    const {sandbox,calls}=setup([null,null]);sandbox.draft={text:'xem giúp',attachments:[{id:'video',occurrenceId:'same-occurrence',path:'C:\\owned.mp4'}]};
    const prepared=await vm.runInContext('attachVideos(ctx,draft)',sandbox);
    await prepared.prepareSession({runtimeSessionId:'owner',storedSessionId:null,submissionId:'stable-request',text:'xem giúp'});
    ids.push(calls[0].body.client_id);
  }
  assert.equal(ids[0],ids[1]);
});
