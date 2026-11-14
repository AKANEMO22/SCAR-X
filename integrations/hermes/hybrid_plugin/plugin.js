import {
  host,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  COMPOSER_AREAS,
  useQuery,
  useValue,
} from "@hermes/plugin-sdk";
import { useEffect, useRef, useState } from "react";
import { jsx, jsxs } from "react/jsx-runtime";

const ID = "scarx-hybrid",
  PATH = "/scarx-hybrid";
const COGNITIVE_LABELS = {
  starting: "Khởi động",
  serving: "Đang làm việc",
  observing: "Đang quan sát",
  reflecting: "Xem lại trải nghiệm",
  verifying: "Kiểm chứng",
  exploring: "Tìm hiểu mục tiêu",
  sleep_consolidating: "Ngủ · củng cố",
  resting: "Nghỉ",
  awake_idle: "Thức · chờ việc",
  paused: "Tạm dừng học",
  waking: "Đang thức dậy",
};
const CSS = `.sxh{height:100%;overflow:auto;padding:22px;max-width:1200px;margin:auto;color:var(--ui-text-primary);font:14px/1.5 system-ui}.sxh h1{font-size:23px;margin:0}.sxh h2{font-size:16px}.sxh small{color:var(--ui-text-secondary)}.sxh button,.sxh input,.sxh textarea{font:inherit;color:inherit;border:1px solid var(--ui-stroke-secondary);border-radius:7px;padding:9px;background:var(--ui-bg-secondary)}.sxh button{cursor:pointer}.sxh textarea{width:100%;min-height:85px}.sxh input{flex:1;min-width:100px}.sxh-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}.sxh-card{padding:15px;border:1px solid var(--ui-stroke-secondary);border-radius:10px;margin:12px 0}.sxh-stat{flex:1;min-width:140px}.sxh-stat b{display:block;font-size:20px}.sxh-feed{max-height:50vh;overflow:auto}.sxh pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace}.sxh-error{color:var(--ui-danger,#ee867e);white-space:pre-wrap}.sxh-message{border-bottom:1px solid var(--ui-stroke-secondary);padding:12px 0;white-space:pre-wrap}.sxh button[aria-selected=true]{border-color:var(--ui-accent,#6cbaac)}`;

function Panel({ ctx }) {
  const [tab, setTab] = useState("trace"),
    [question, setQuestion] = useState(""),
    [stream, setStream] = useState(""),
    [video, setVideo] = useState(""),
    [topic, setTopic] = useState(""),
    [reply, setReply] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [history, setHistory] = useState([]),
    [trace, setTrace] = useState([]),
    [checkpoint, setCheckpoint] = useState(""),
    [archive, setArchive] = useState(null),
    cursor = useRef(0),
    traceProcess = useRef(null),
    lastInteraction = useRef(Date.now());
  const query = useQuery({
      queryKey: [ID, "status"],
      queryFn: () => ctx.rest("/status?after=" + cursor.current),
      refetchInterval: 1800,
      retry: 1,
    }),
    s = query.data;
  useEffect(() => {
    if (!s) return;
    if (traceProcess.current !== null && traceProcess.current !== s.pid) {
      cursor.current = 0;
      setTrace([]);
      setHistory([]);
      traceProcess.current = s.pid;
      return;
    }
    traceProcess.current = s.pid;
    let rows = s.events || [];
    if (rows.length) {
      cursor.current = rows.at(-1).id;
      setTrace((v) => [...v, ...rows].slice(-150));
      let answers = rows
        .filter((e) => e.kind === "answer")
        .map((e) => e.payload);
      if (answers.length) setHistory((v) => [...v, ...answers].slice(-60));
    }
  }, [s]);
  useEffect(() => {
    let lastActivityPing = 0;
    const mark = () => {
      lastInteraction.current = Date.now();
      if (Date.now() - lastActivityPing >= 2000) {
        lastActivityPing = Date.now();
        void ctx
          .rest("/presence", { method: "POST", body: { activity: true } })
          .catch(() => {});
      }
    };
    const presence = () => {
      if (
        document.visibilityState === "visible" &&
        document.hasFocus() &&
        Date.now() - lastInteraction.current < 30000
      )
        void ctx
          .rest("/presence", { method: "POST", body: { activity: false } })
          .catch(() => {});
    };
    const disposers = [
      ctx.addEventListener(window, "pointerdown", mark),
      ctx.addEventListener(window, "keydown", mark),
      ctx.setInterval(presence, 15000),
    ];
    presence();
    return () => disposers.forEach((d) => d());
  }, [ctx]);
  async function send(path, body) {
    setBusy(true);
    setError("");
    try {
      let result = await ctx.rest(path, {
        method: "POST",
        body: { ...body, client_id: crypto.randomUUID() },
        timeoutMs: 120000,
      });
      void query.refetch();
      return result;
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setBusy(false);
    }
  }
  const button = (text, click, extra = {}) =>
    jsx("button", {
      type: "button",
      disabled: busy,
      onClick: click,
      ...extra,
      children: text,
    });
  const stat = (label, value) =>
    jsxs("div", {
      className: "sxh-card sxh-stat",
      children: [
        jsx("small", { children: label }),
        jsx("b", { children: value }),
      ],
    });
  return jsxs("section", {
    className: "sxh",
    children: [
      jsx("h1", { children: "SCAR-X · Debug" }),
      jsx("small", {
        children:
          "Trò chuyện ở chat Hermes bình thường với model SCAR-X Hybrid. Trang này chỉ theo dõi hoạt động và học.",
      }),
      jsxs("div", {
        className: "sxh-row",
        children: [
          stat(
            "RAM toàn hệ",
            s?.resources
              ? `${((s.resources.conservative_ram_bytes ?? s.resources.rss_bytes) / 1e6).toFixed(0)} MB · ${s.resources.limit_bytes == null ? "GPU · không đặt trần" : `CPU · trần ${(s.resources.limit_bytes / 1e6).toFixed(0)} MB`}`
              : "Đang kết nối",
          ),
          stat(
            "VRAM riêng · nhóm tiến trình",
            s?.resources?.gpu_dedicated?.available
              ? `${(s.resources.gpu_dedicated.bytes / 1048576).toFixed(0)} MiB`
              : "Chưa đo",
          ),
          stat(
            "Neuron phát xung / đã cấp",
            s?.neural ? `${s.neural.active} / ${s.neural.hidden}` : "—",
          ),
          stat("Cập nhật synapse", s?.neural?.updates ?? "—"),
          stat("Lượt xem lại đã hoàn tất", s?.review?.completed ?? "—"),
        ],
      }),
      jsx("p", { children: s?.phase || "Đang kết nối gateway…" }),
      s?.cognition &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("h2", { children: COGNITIVE_LABELS[s.cognition.phase] || s.cognition.phase }),
            jsx("p", { children: s.cognition.reason }),
            s.execution_placement && jsx("p", {
              children: `Tính toán: VLM/vision ${s.execution_placement.VLM_vision.toUpperCase()} · ASR ${s.execution_placement.ASR.toUpperCase()} · ALIF ${s.execution_placement.ALIF_forward_learning.toUpperCase()}. I/O và điều phối trên CPU.`,
            }),
            jsx("small", {
              children: `VLM ${s.cognition.VLM_resident ? "đang giữ tài nguyên RAM/VRAM" : "đã nhả tài nguyên"} · ${s.cognition.state.consolidations} lượt ôn có nguồn · ${s.cognition.state.replay_used}/${s.config.sleep_replay_budget} ngân sách lượt hiện tại. Đây là trạng thái xử lý, không phải độc thoại nội tâm.`,
            }),
            jsxs("div", {
              className: "sxh-row",
              children: [
                ["auto", "Tự động"], ["awake", "Thức"], ["sleep", "Ngủ / củng cố"],
              ].map(([intent, label]) => button(label, () => { void send("/cognition", { intent }).catch(() => {}); }, {
                "aria-pressed": s.cognition.state.intent === intent,
              })),
            }),
          ],
        }),
      s?.review &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("b", { children: "Tự xem lại công việc" }),
            jsx("p", {
              children: `${s.review.pending} lượt chờ · ${s.review.completed} đã xem lại · ${s.review.verified_replay_items} mẫu có nguồn để ôn · ${s.review.replay_updates} cập nhật replay`,
            }),
            jsx("small", {
              children:
                "Trạng thái: " +
                (s.review.blocker || "sẵn sàng") +
                " · VLM frozen; review chưa được kiểm chứng không tự trở thành nhãn đúng.",
            }),
            s.review.last_result &&
              jsx("pre", {
                children: JSON.stringify(s.review.last_result, null, 2),
              }),
          ],
        }),
      (error || query.error || s?.failure) &&
        jsx("p", {
          className: "sxh-error",
          role: "alert",
          children: error || query.error?.message || s?.failure,
        }),
      jsxs("nav", {
        className: "sxh-row",
        children: [
          ["video", "Trí nhớ video"],
          ["learn", "Hỏi để học"],
          ["trace", "Trace trực tiếp"],
        ].map(([id, label]) =>
          button(label, () => setTab(id), { "aria-selected": tab === id }),
        ),
      }),
      tab === "chat" &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("div", {
              className: "sxh-feed",
              children: history.length
                ? history.map((a) =>
                    jsxs(
                      "article",
                      {
                        className: "sxh-message",
                        children: [
                          jsx("b", { children: a.question }),
                          jsx("p", { children: a.answer }),
                          ...(a.evidence || []).map((e) =>
                            jsx("small", {
                              children: `[${e.id}] ${e.start.toFixed(1)}–${e.end.toFixed(1)}s · ${e.source}`,
                            }),
                          ),
                        ],
                      },
                      a.id,
                    ),
                  )
                : jsx("small", {
                    children:
                      "Bạn có thể trò chuyện hoặc hỏi về video đã phân tích.",
                  }),
            }),
            jsx("textarea", {
              value: question,
              maxLength: 6000,
              onChange: (e) => setQuestion(e.target.value),
              placeholder: "Nhắn cho SCAR-X…",
            }),
            jsxs("div", {
              className: "sxh-row",
              children: [
                jsx("input", {
                  value: stream,
                  onChange: (e) => setStream(e.target.value),
                  placeholder: "ID video (trống = trò chuyện chung)",
                }),
                button(
                  busy ? "Đang trả lời…" : "Gửi",
                  () =>
                    void send("/chat", {
                      messages: [{ role: "user", content: question }],
                      scarx_stream_id: stream || null,
                    })
                      .then(() => setQuestion(""))
                      .catch(() => {}),
                  { disabled: busy || !s?.ready || !question.trim() },
                ),
              ],
            }),
          ],
        }),
      tab === "video" &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("h2", { children: "Trí nhớ video" }),
            jsx("small", {
              children:
                "Đính kèm video hoặc chọn cửa sổ từ ô chat Hermes. Đây là tiến độ thực, không đồng nghĩa toàn bộ video đã được hiểu.",
            }),
            ...(s?.streams || []).map((v) =>
              jsxs(
                "div",
                {
                  className: "sxh-card",
                  children: [
                    jsx("b", { children: v.id + " · " + v.status + (v.degraded ? " · phân tích chưa đầy đủ" : "") }),
                    jsx("p", {
                      children: `Đã thu ${v.capture_frontier.toFixed(1)}s · âm thanh ${v.asr_frontier.toFixed(1)}s · ảnh ${v.vision_frontier.toFixed(1)}s`,
                    }),
                    jsx("small", {
                      children:
                        "Nguồn đã lưu để debug; không tự gắn vào tin nhắn.",
                    }),
                  ],
                },
                v.id,
              ),
            ),
          ],
        }),
      tab === "learn" &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("h2", { children: "Hỏi để học" }),
            jsx("small", {
              children:
                "Khi bạn vắng: tự tìm hiểu, gom câu hỏi, không thông báo. Bỏ qua không thành nhãn sai.",
            }),
            ...(s?.tool_incidents?.recent || []).map((incident) => jsxs("div", {
              className: "sxh-card",
              children: [
                jsx("b", { children: `Lỗi đang kiểm tra: ${incident.tool} lặp ${incident.observed_repeats} lần cùng kết quả` }),
                jsx("p", { children: "Nguyên nhân chưa được xác minh. Việc phát hiện lặp chưa phải bằng chứng synapse đã học cách sửa." }),
                jsx("small", { children: `${incident.status} · ${incident.observed_empty_result ? "Kết quả rỗng" : "Kết quả không thay đổi"}` }),
                incident.diagnostic_probe && jsx("p", { children: `Đã kiểm tra ${incident.diagnostic_probe.source_name} trong project ${incident.diagnostic_probe.project_label}: ${incident.diagnostic_probe.read_ok ? "đọc được nguồn thay thế; cần xác nhận đúng ý bạn" : "chưa xác lập được nguồn đọc được"}.` }),
                incident.last_policy_action && jsx("p", { children: `Lựa chọn gần nhất: ${incident.last_policy_action.action} · ${incident.last_policy_action.selector}` }),
                incident.last_feedback && jsx("pre", { children: JSON.stringify(incident.last_feedback, null, 2) }),
              ],
            }, incident.id)),
            s?.question &&
              jsxs("div", {
                className: "sxh-card",
                children: [
                  jsx("b", { children: s.question.question }),
                  jsx("textarea", {
                    value: reply,
                    onChange: (e) => setReply(e.target.value),
                    placeholder: "Trả lời…",
                  }),
                  jsxs("div", {
                    className: "sxh-row",
                    children: [
                      button(
                        "Trả lời",
                        () =>
                          void send("/feedback", {
                            id: s.question.id,
                            text: reply,
                          })
                            .then(() => setReply(""))
                            .catch(() => {}),
                      ),
                      button("Trả lời · hữu ích", () => void send("/feedback", {
                        id: s.question.id,
                        text: reply,
                        helpful: true,
                      }).then(() => setReply("")).catch(() => {})),
                      button(
                        "Bỏ qua",
                        () =>
                          void send("/feedback", {
                            id: s.question.id,
                            text: "bỏ qua",
                          }).catch(() => {}),
                      ),
                      button("Chưa hữu ích", () => void send("/feedback", {
                        id: s.question.id,
                        text: "Câu hỏi này chưa hữu ích cho tác vụ.",
                        helpful: false,
                      }).catch(() => {})),
                    ],
                  }),
                ],
              }),
            ...(s?.goals || [])
              .slice(-15)
              .reverse()
              .map((g) =>
                jsxs(
                  "div",
                  {
                    className: "sxh-message",
                    children: [
                      jsx("b", { children: g.question }),
                      jsx("p", {
                        children: `${g.status} · ${g.attempts}/2 vòng · ${g.last_action || "chờ"}`,
                      }),
                    ],
                  },
                  g.id,
                ),
              ),
            jsxs("div", {
              className: "sxh-row",
              children: [
                jsx("input", {
                  value: topic,
                  maxLength: 160,
                  onChange: (e) => setTopic(e.target.value),
                  placeholder: "Chủ đề công khai cho phép tự tra web",
                }),
                button(
                  "Cho phép chủ đề",
                  () =>
                    void send("/public-topic", { topic })
                      .then(() => setTopic(""))
                      .catch(() => {}),
                ),
              ],
            }),
            ...(s?.public_topics || []).map((t) =>
              jsx("small", { children: t.topic + " · " + t.id }, t.id),
            ),
            jsxs("div", {
              className: "sxh-row",
              children: [
                button(
                  s?.learning_enabled ? "Tạm dừng học" : "Tiếp tục học",
                  () =>
                    void send("/learning", {
                      enabled: !s?.learning_enabled,
                    }).catch(() => {}),
                ),
                button(
                  "Lưu trạng thái",
                  () =>
                    void send("/save", {})
                      .then((r) => setCheckpoint(r.id))
                      .catch(() => {}),
                ),
                jsx("input", {
                  value: checkpoint,
                  onChange: (e) => setCheckpoint(e.target.value),
                  placeholder: "ID checkpoint",
                }),
                button(
                  "Khôi phục",
                  () =>
                    void send("/rollback", { id: checkpoint }).catch(() => {}),
                ),
              ],
            }),
          ],
        }),
      tab === "trace" &&
        jsxs("div", {
          className: "sxh-card",
          children: [
            jsx("h2", { children: "Trace trực tiếp" }),
            jsx("small", {
              children:
                "Nhật ký lưu trên SSD, độc lập với cửa sổ token. Trang trực tiếp chỉ giữ một phần gần nhất trong RAM.",
            }),
            jsxs("div", {
              className: "sxh-row",
              children: [
                button(
                  "Xem nhật ký từ đầu",
                  () =>
                    void ctx
                      .rest("/status?after=0")
                      .then((r) => setArchive(r.events || []))
                      .catch((e) => setError(e.message)),
                ),
                archive &&
                  button(
                    "Trang tiếp theo",
                    () =>
                      void ctx
                        .rest("/status?after=" + (archive.at(-1)?.id || 0))
                        .then((r) => setArchive(r.events || []))
                        .catch((e) => setError(e.message)),
                  ),
                archive && button("Đóng lịch sử", () => setArchive(null)),
              ],
            }),
            archive &&
              jsxs("div", {
                className: "sxh-card sxh-feed",
                children: [
                  jsx("small", {
                    children: archive.length
                      ? `Sự kiện ${archive[0].id}–${archive.at(-1).id}`
                      : "Không có sự kiện tiếp theo",
                  }),
                  ...archive.map((e) =>
                    jsxs(
                      "details",
                      {
                        children: [
                          jsx("summary", {
                            children:
                              e.id +
                              " · " +
                              new Date(e.created * 1000).toLocaleString() +
                              " · " +
                              e.kind,
                          }),
                          jsx("pre", {
                            children: JSON.stringify(e.payload, null, 2),
                          }),
                        ],
                      },
                      "archive" + e.id,
                    ),
                  ),
                ],
              }),
            jsx("small", {
              children:
                "Tự cập nhật hoạt động thực; actor phân biệt neural, model và thao tác triển khai.",
            }),
            jsx("div", {
              className: "sxh-feed",
              children: [...trace].reverse().map((e) =>
                jsxs(
                  "details",
                  {
                    children: [
                      jsx("summary", {
                        children:
                          new Date(e.created * 1000).toLocaleTimeString() +
                          " · " +
                          e.actor +
                          " · " +
                          e.kind,
                      }),
                      jsx("pre", {
                        children: JSON.stringify(e.payload, null, 2),
                      }),
                    ],
                  },
                  e.id,
                ),
              ),
            }),
          ],
        }),
    ],
  });
}

function sessionIds() {
  return [
    ...new Set(
      [
        host.state.focusedSessionId.get(),
        host.state.focusedStoredSessionId.get(),
      ].filter((id) => typeof id === "string" && id && id !== "draft"),
    ),
  ];
}
function clearLegacyVideoContext(ctx) {
  for (const key of ["draft", ...sessionIds()])
    ctx.storage.remove("video:" + key);
}
const pendingUploads = new Map();
function isScarx() {
  return String(host.state.model.get() || "").includes("scarx-hybrid");
}
async function attachVideos(ctx, draft) {
  if (!isScarx()) return draft;
  clearLegacyVideoContext(ctx);
  const videos = (draft.attachments || []).filter(
    (a) => a.path && /\.(mp4|mkv|webm|mov|avi)$/i.test(a.path),
  );
  if (videos.length > 1) {
    host.notifyError("Bản đầu nhận một video mỗi lượt.");
    return null;
  }
  // Plain turns remain byte-for-byte the same draft, regardless of old plugin
  // storage. No sticky global "draft" video or renderer-authored prompt suffix.
  if (!videos.length) return draft;
  const sessions = sessionIds();
  try {
    for (const attachment of videos) {
      const key = JSON.stringify([
        sessions,
        attachment.occurrenceId || attachment.id || attachment.path,
      ]);
      let upload = pendingUploads.get(key);
      if (!upload || Date.now() - upload.created > 240000) {
        upload = { id: crypto.randomUUID(), created: Date.now(), stream: null };
        pendingUploads.set(key, upload);
        while (pendingUploads.size > 16)
          pendingUploads.delete(pendingUploads.keys().next().value);
      }
      if (!upload.stream)
        upload.stream = await ctx.rest("/observe", {
          method: "POST",
          body: { path: attachment.path, client_id: upload.id },
          timeoutMs: 20000,
        });
      if (JSON.stringify(sessionIds()) !== JSON.stringify(sessions))
        throw Error(
          "Đã chuyển phiên chat; video chưa được gắn. Gửi lại ở đúng cuộc trò chuyện.",
        );
      const bytes = new TextEncoder().encode(draft.text.trim());
      const digest = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        (v) => v.toString(16).padStart(2, "0"),
      ).join("");
      if (upload.digest !== digest) {
        upload.ticket = crypto.randomUUID();
        upload.digest = digest;
      }
      await ctx.rest("/observe", {
        method: "POST",
        body: {
          existing_stream_id: upload.stream.id,
          video_context: {
            text_sha256: digest,
            session_ids: sessions,
            ticket_id: upload.ticket,
          },
        },
        timeoutMs: 15000,
      });
    }
    return {
      ...draft,
      attachments: (draft.attachments || []).filter((a) => !videos.includes(a)),
    };
  } catch (e) {
    host.notifyError(e.message);
    return null;
  }
}
const QUESTION_CSS = ".sxq{box-sizing:border-box;display:grid;gap:12px;width:100%;min-width:0;padding:14px;margin:0 0 10px;border:1px solid var(--ui-stroke-secondary,#343b43);border-radius:12px;background:var(--ui-bg-secondary,#151a20);color:var(--ui-text-primary,#e4e8ec);font:14px/1.5 system-ui,sans-serif}.sxq *{box-sizing:border-box}.sxq-header{display:flex;align-items:center;gap:8px;color:var(--ui-text-secondary,#a3abb3);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.sxq-dot{width:6px;height:6px;flex:none;border-radius:50%;background:var(--ui-accent,#80c9b0)}.sxq-question{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;font-weight:500;line-height:1.55}.sxq textarea{display:block;width:100%;min-width:0;min-height:64px;margin:0;padding:10px 12px;resize:vertical;border:1px solid var(--ui-stroke-secondary,#343b43);border-radius:8px;background:var(--ui-bg-primary,#10141a);color:inherit;font:inherit;outline:none}.sxq textarea::placeholder{color:var(--ui-text-tertiary,#858e97)}.sxq textarea:focus-visible{border-color:var(--ui-accent,#80c9b0);box-shadow:0 0 0 2px color-mix(in srgb,var(--ui-accent,#80c9b0) 16%,transparent)}.sxq-footer{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.sxq-actions,.sxq-rating{display:flex;gap:6px;align-items:center}.sxq-rating{color:var(--ui-text-secondary,#a3abb3);font-size:12px}.sxq button{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:32px;margin:0;padding:6px 11px;border:1px solid transparent;border-radius:7px;background:transparent;color:inherit;font:inherit;white-space:nowrap;cursor:pointer}.sxq button:hover{background:var(--ui-bg-tertiary,#242c34)}.sxq button:focus-visible{outline:2px solid var(--ui-accent,#80c9b0);outline-offset:2px}.sxq button:disabled{opacity:.45;cursor:default}.sxq .sxq-rate{width:30px;height:30px;min-height:30px;padding:6px}.sxq .sxq-rate[aria-pressed=true]{color:var(--ui-accent,#80c9b0);background:color-mix(in srgb,var(--ui-accent,#80c9b0) 12%,transparent);border-color:var(--ui-stroke-secondary,#343b43)}.sxq .sxq-skip{color:var(--ui-text-secondary,#a3abb3)}.sxq .sxq-submit{background:var(--ui-accent,#80c9b0);color:var(--ui-on-accent,#10261e);font-weight:600}.sxq .sxq-submit:hover{filter:brightness(1.08)}@media(max-width:520px){.sxq{padding:12px;gap:10px}.sxq-footer{align-items:flex-start}.sxq-actions{margin-left:auto}.sxq-rating-label{display:none}}";

function questionRatingIcon(positive) {
  return jsxs("svg", {
    width: 16, height: 16, viewBox: "0 0 24 24", fill: "none",
    stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round",
    strokeLinejoin: "round", "aria-hidden": true,
    style: positive ? undefined : { transform: "rotate(180deg)" },
    children: [
      jsx("path", { d: "M7 10v10H3V10z" }),
      jsx("path", { d: "M7 19h10a2 2 0 0 0 2-1.7l1-6A2 2 0 0 0 18 9h-5l1-4a2 2 0 0 0-2-2L7 10" }),
    ],
  });
}

function LearningQuestion({ ctx }) {
  const model = useValue(host.state.model),
    [reply, setReply] = useState(""),
    [busy, setBusy] = useState(false),
    [rating, setRating] = useState(null);
  const q = useQuery({
    queryKey: [ID, "pending-question"],
    queryFn: () => ctx.rest("/status?after=2147483647"),
    refetchInterval: 8000,
    retry: 1,
    enabled: String(model || "").includes("scarx-hybrid"),
  });
  const pendingId = q.data?.question?.id;
  useEffect(() => {
    setReply("");
    setRating(null);
  }, [pendingId]);
  useEffect(() => {
    let last = 0;
    const ping = () => {
      if (!isScarx() || Date.now() - last < 15000) return;
      last = Date.now();
      void ctx
        .rest("/presence", { method: "POST", body: { activity: true } })
        .catch(() => {});
    };
    const a = ctx.addEventListener(window, "pointerdown", ping),
      b = ctx.addEventListener(window, "keydown", ping);
    return () => {
      a();
      b();
    };
  }, [ctx]);
  if (!String(model || "").includes("scarx-hybrid") || !q.data?.question)
    return null;
  const question = q.data.question;
  async function answer(text, helpful) {
    setBusy(true);
    try {
      await ctx.rest("/feedback", {
        method: "POST",
        body: { id: question.id, text,
          ...(typeof helpful === "boolean" ? { helpful } : {}) },
      });
      setReply("");
      setRating(null);
      void q.refetch();
    } catch (e) {
      host.notifyError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const canSend = !busy && (!!reply.trim() || rating === false);
  const submit = () => {
    if (!canSend) return;
    void answer(reply.trim() || "Câu hỏi này chưa hữu ích cho tác vụ.", rating);
  };
  return jsxs("form", {
    className: "sxq",
    "aria-label": "Hỏi để học từ SCAR-X",
    onSubmit: (event) => { event.preventDefault(); submit(); },
    children: [
      jsxs("div", { className: "sxq-header", children: [
        jsx("span", { className: "sxq-dot", "aria-hidden": true }),
        jsx("span", { children: "SCAR-X · Hỏi để học" }),
      ]}),
      jsx("p", { className: "sxq-question", children: question.question }),
      jsx("textarea", {
        value: reply,
        onChange: (e) => setReply(e.target.value),
        onKeyDown: (event) => {
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            submit();
          }
        },
        rows: 2,
        maxLength: 4000,
        disabled: busy,
        "aria-label": "Trả lời SCAR-X",
        placeholder: "Nhập câu trả lời của bạn…",
      }),
      jsxs("div", { className: "sxq-footer", children: [
        jsxs("div", { className: "sxq-rating", children: [
          jsx("span", { className: "sxq-rating-label", children: "Câu hỏi hữu ích?" }),
          ...[[true, "Hữu ích"], [false, "Chưa hữu ích"]].map(([value, label]) =>
            jsx("button", {
              type: "button", className: "sxq-rate", title: label,
              "aria-label": label, "aria-pressed": rating === value,
              disabled: busy,
              onClick: () => setRating(rating === value ? null : value),
              children: questionRatingIcon(value),
            }, label)),
        ]}),
        jsxs("div", { className: "sxq-actions", children: [
          jsx("button", {
            type: "button", className: "sxq-skip", disabled: busy,
            onClick: () => void answer("bỏ qua"), children: "Bỏ qua",
          }),
          jsx("button", {
            type: "submit", className: "sxq-submit", disabled: !canSend,
            children: busy ? "Đang gửi…" : (!reply.trim() && rating === false ? "Gửi đánh giá" : "Gửi trả lời"),
          }),
        ]}),
      ]}),
    ],
  });
}

export default {
  id: ID,
  name: "SCAR-X Debug & video input",
  description:
    "Normal Hermes model uses one local gateway; this panel is observation only.",
  register(ctx) {
    clearLegacyVideoContext(ctx);
    const style = document.createElement("style");
    style.textContent = CSS + QUESTION_CSS;
    document.head.append(style);
    ctx.onDispose(() => style.remove());
    ctx.registerMany([
      {
        id: "page",
        area: ROUTES_AREA,
        title: "SCAR-X Debug",
        data: { path: PATH },
        render: () => jsx(Panel, { ctx }),
      },
      {
        id: "nav",
        area: SIDEBAR_NAV_AREA,
        order: 59,
        data: { codicon: "hubot", label: "SCAR-X Debug", path: PATH },
      },
      {
        id: "question",
        area: COMPOSER_AREAS.top,
        render: () => jsx(LearningQuestion, { ctx }),
      },
      {
        id: "video-attachments",
        area: COMPOSER_AREAS.middleware,
        order: 10,
        data: { handler: (draft) => attachVideos(ctx, draft) },
      },
    ]);
  },
};
