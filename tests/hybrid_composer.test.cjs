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
    host: {
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
      return endpoint === "/observe"
        ? { id: "a".repeat(32) }
        : { queued: true };
    },
  };
  return { sandbox, calls, storage };
}

test("learning replies reward the controller only with an explicit rating", async () => {
  for (const [label, rating] of [
    ["Trả lời", undefined],
    ["Trả lời · hữu ích", true],
    ["Chưa hữu ích", false],
    ["Bỏ qua", undefined],
  ]) {
    const { sandbox, calls } = setup();
    sandbox.useValue = (atom) => atom.get();
    sandbox.useState = (initial) => [initial === "" ? "Owner guidance" : initial, () => {}];
    sandbox.useQuery = () => ({
      data: { question: { id: "pending" } },
      refetch: async () => {},
    });
    sandbox.useEffect = () => {};
    sandbox.jsx = sandbox.jsxs = (tag, props) => ({ tag, ...props });
    const widget = vm.runInContext("LearningQuestion({ctx})", sandbox);
    const button = widget.children.find((c) => c.tag === "button" && c.children === label);
    assert.ok(button);
    button.onClick();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls.length, 1);
    assert.equal(calls[0].endpoint, "/feedback");
    assert.equal(calls[0].body.helpful, rating);
    assert.equal(Object.hasOwn(calls[0].body, "helpful"), typeof rating === "boolean");
  }
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
  assert.deepEqual(
    calls.map((c) => c.endpoint),
    ["/observe", "/observe"],
  );
  assert.equal(calls[1].body.existing_stream_id, "a".repeat(32));
  assert.match(calls[1].body.video_context.text_sha256, /^[a-f0-9]{64}$/);
  assert.equal(
    JSON.stringify(calls[1].body.video_context.session_ids),
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
  assert.equal(calls[1].body.video_context.session_ids.length, 0);
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
  assert.equal(await vm.runInContext("attachVideos(ctx,draft)", sandbox), null);
  assert.equal(JSON.stringify(sandbox.draft), before);
  assert.match(calls[0].error, /upload failure/);
});
