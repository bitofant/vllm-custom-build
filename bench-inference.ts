#!/usr/bin/env -S node
/**
 * Quick inference benchmark against a running vLLM OpenAI-compatible server.
 *
 * Streams a handful of varied prompts through /v1/chat/completions, measuring
 * per-request time-to-first-token (TTFT) and decode throughput (tok/s). One
 * throwaway request runs first and is discarded: the first prompt of a run pays
 * a warmup cost worth more than every difference this benchmark exists to
 * detect. Records written before 2026-09-06 lack it (`warmup` field absent) and
 * their first row reads high.
 *
 * Note the prompts are ~60-80 tokens, so TTFT here is per-request overhead, not
 * prefill speed — there is not enough prompt to measure prefill. Measuring that
 * would need a multi-thousand-token prompt, and then prefix caching WOULD skew
 * it (see WARMUP_PROMPT) and would have to be defeated per run. When the
 * server has speculative decoding enabled (our Gemma 4 `4m` MTP config), it also
 * scrapes the Prometheus /metrics endpoint before/after to report draft
 * acceptance rate and mean accept length.
 *
 * Every run appends one JSON record to `bench-results/bench.jsonl` (gitignored)
 * so numbers can be compared across builds; `--report` renders that history as a
 * table instead of running a benchmark.
 *
 * Timestamps are stored as UTC (`ts`, ISO-8601) but DISPLAYED in the local
 * timezone, so `--report` lines up with local-time logs such as
 * ~/scripts/vllm-startups.csv. Keep that split: don't localize what's written
 * to disk, and don't print `ts` raw.
 *
 * No dependencies — uses Node's native fetch + TypeScript type-stripping
 * (Node >= 23.6, or 22.x with --experimental-strip-types). Run directly:
 *
 *     ./bench-inference.ts
 *     node bench-inference.ts --host host:8000 --max-tokens 256
 *     node bench-inference.ts --model RedHatAI/gemma-4-31B-it-NVFP4
 *     node bench-inference.ts --tag b17-baseline     # label this run
 *     node bench-inference.ts --no-save              # don't append to history
 *     node bench-inference.ts --report               # print saved history
 */

import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Resolve next to the script, so results land here regardless of cwd.
const RESULTS_DIR = join(dirname(fileURLToPath(import.meta.url)), "bench-results");
const RESULTS_FILE = join(RESULTS_DIR, "bench.jsonl");

// Discarded first request. The first prompt of a run pays a warmup cost that has
// nothing to do with the build under test: measured 2026-09-06, two runs six
// seconds apart gave short-factual TTFT 0.741 then 0.046 (and code 0.461 then
// 0.063) while the untouched later rows sat at 0.065 both times. Whichever
// prompt goes first eats that tax, so spend it on a throwaway.
//
// Deliberately NOT one of the PROMPTS below: on configs with an ordinary block
// size (gemma `4`: 16) a ~70-token prompt is several full blocks, so reusing a
// measured prompt here would prime the prefix cache for its own measurement.
// (On the hybrid configs block_size is 1616 and nothing under 1616 tokens is
// ever cached, but don't rely on that holding for every model.)
const WARMUP_PROMPT = "Reply with the single word: ready.";
const WARMUP_TOKENS = 32;

// (label, prompt) — a spread of decode-heavy tasks so tok/s is meaningful.
const PROMPTS: [string, string][] = [
  ["short-factual", "In one sentence, what is the capital of Australia?"],
  ["code", "Write a Python function that returns the nth Fibonacci number iteratively. Code only."],
  ["reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Explain your reasoning step by step."],
  ["long-prose", "Write a detailed 200-word explanation of how a transformer neural network attention mechanism works."],
];

interface Args {
  host: string;
  model: string | null;
  maxTokens: number;
  tag: string | null;
  save: boolean;
  report: boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    host: "localhost:8000",
    model: null,
    maxTokens: 256,
    tag: null,
    save: true,
    report: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--host") args.host = argv[++i];
    else if (a === "--model") args.model = argv[++i];
    else if (a === "--max-tokens") args.maxTokens = parseInt(argv[++i], 10);
    else if (a === "--tag") args.tag = argv[++i];
    else if (a === "--no-save") args.save = false;
    else if (a === "--report") args.report = true;
    else if (a === "-h" || a === "--help") {
      console.log(
        "usage: bench-inference.ts [--host H:P] [--model ID] [--max-tokens N]\n" +
        "                          [--tag LABEL] [--no-save] [--report]\n\n" +
        `  results are appended to ${RESULTS_FILE}\n` +
        "  --tag LABEL   label the run (e.g. the build number under test)\n" +
        "  --no-save     run the benchmark but don't append to history\n" +
        "  --report      print saved history instead of benchmarking",
      );
      process.exit(0);
    }
  }
  return args;
}

async function detectModel(host: string): Promise<string> {
  const r = await fetch(`http://${host}/v1/models`);
  const data = await r.json();
  return data.data[0].id;
}

// vLLM serves its own version at /version; other OpenAI-compatible servers
// (llama.cpp, ...) 404 here, which is recorded as an unknown version, not an error.
async function detectServerVersion(host: string): Promise<string | null> {
  try {
    const r = await fetch(`http://${host}/version`);
    if (!r.ok) return null;
    const data = await r.json();
    return typeof data?.version === "string" ? data.version : null;
  } catch {
    return null;
  }
}

interface SpecMetrics {
  drafts: number;
  draftTokens: number;
  accepted: number;
}

async function scrapeSpecMetrics(host: string): Promise<SpecMetrics | null> {
  let text: string;
  try {
    const r = await fetch(`http://${host}/metrics`);
    text = await r.text();
  } catch {
    return null;
  }
  const grab = (metric: string): number | null => {
    const re = new RegExp(`^${metric.replace(/[:]/g, "\\$&")}\\{[^}]*}\\s+([0-9.eE+-]+)`, "m");
    const m = text.match(re);
    return m ? parseFloat(m[1]) : null;
  };
  const drafts = grab("vllm:spec_decode_num_drafts_total");
  const draftTokens = grab("vllm:spec_decode_num_draft_tokens_total");
  const accepted = grab("vllm:spec_decode_num_accepted_tokens_total");
  if (drafts === null || draftTokens === null || accepted === null) return null;
  return { drafts, draftTokens, accepted };
}

interface Result {
  label: string;
  ttft: number | null;
  total: number;
  completionTokens: number;
  decodeTps: number;
  totalTps: number;
  sawContent: boolean;
  finishReason: string | null;
}

// Reasoning models (Qwen3.x, gpt-oss, ...) stream their thinking as `reasoning`
// or `reasoning_content` deltas and may never emit a `content` delta at all.
// Those are decoded tokens too, so TTFT is whichever text arrives first.
function deltaText(delta: any): string {
  return delta?.content || delta?.reasoning || delta?.reasoning_content || "";
}

async function runPrompt(host: string, model: string, prompt: string, maxTokens: number): Promise<Result> {
  const body = JSON.stringify({
    model,
    messages: [{ role: "user", content: prompt }],
    max_tokens: maxTokens,
    temperature: 0.0,
    stream: true,
    stream_options: { include_usage: true },
  });
  const tStart = performance.now();
  let ttft: number | null = null;
  let completionTokens = 0;
  let sawContent = false;
  let finishReason: string | null = null;

  const resp = await fetch(`http://${host}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      const chunk = JSON.parse(payload);
      const choice = chunk.choices?.[0];
      if (deltaText(choice?.delta)) {
        if (ttft === null) ttft = (performance.now() - tStart) / 1000;
        if (choice.delta.content) sawContent = true;
      }
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (chunk.usage?.completion_tokens != null) {
        completionTokens = chunk.usage.completion_tokens;
      }
    }
  }

  const total = (performance.now() - tStart) / 1000;
  // No clamped floor here: if TTFT is missing the decode window is unknowable,
  // and a fabricated 0 tok/s reads as broken rather than as a 1e9 tok/s "result".
  const decodeTime = ttft === null ? 0 : total - ttft;
  const decodeTps =
    ttft !== null && completionTokens > 1 && decodeTime > 0
      ? (completionTokens - 1) / decodeTime
      : 0;
  const totalTps = total > 0 ? completionTokens / total : 0;
  return { label: "", ttft, total, completionTokens, decodeTps, totalTps, sawContent, finishReason };
}

function pad(s: string | number, w: number): string {
  return String(s).padStart(w);
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

interface Aggregate {
  decodeTpsMean: number | null;
  decodeTpsMedian: number | null;
  decodeTpsMin: number | null;
  decodeTpsMax: number | null;
  ttftMean: number | null;
  ttftMedian: number | null;
  completionTokens: number;
}

interface SpecSummary {
  drafts: number;
  draftTokens: number;
  accepted: number;
  acceptRate: number;
  acceptLen: number;
}

interface BenchRecord {
  ts: string;
  tag: string | null;
  host: string;
  model: string;
  serverVersion: string | null;
  maxTokens: number;
  /**
   * Whether a throwaway request preceded the measured prompts. Absent on
   * records written before 2026-09-06; those have an inflated first row
   * (usually short-factual) and their TTFTs are not comparable to warmed runs.
   */
  warmup: boolean;
  results: Result[];
  aggregate: Aggregate;
  spec: SpecSummary | null;
}

function summarize(results: Result[]): Aggregate {
  const decs = results.map((r) => r.decodeTps).filter((x) => x > 0);
  const ttfts = results.map((r) => r.ttft).filter((x): x is number => x !== null);
  return {
    decodeTpsMean: decs.length ? mean(decs) : null,
    decodeTpsMedian: decs.length ? median(decs) : null,
    decodeTpsMin: decs.length ? Math.min(...decs) : null,
    decodeTpsMax: decs.length ? Math.max(...decs) : null,
    ttftMean: ttfts.length ? mean(ttfts) : null,
    ttftMedian: ttfts.length ? median(ttfts) : null,
    completionTokens: results.reduce((a, r) => a + r.completionTokens, 0),
  };
}

function saveRecord(rec: BenchRecord): void {
  mkdirSync(RESULTS_DIR, { recursive: true });
  appendFileSync(RESULTS_FILE, JSON.stringify(rec) + "\n");
}

function loadRecords(): BenchRecord[] {
  let raw: string;
  try {
    raw = readFileSync(RESULTS_FILE, "utf8");
  } catch {
    return [];
  }
  // Skip unparseable lines rather than dying: a run killed mid-append should
  // never make the whole history unreadable.
  return raw
    .split("\n")
    .filter((l) => l.trim())
    .flatMap((l) => {
      try {
        return [JSON.parse(l) as BenchRecord];
      } catch {
        return [];
      }
    });
}

function fmt(x: number | null | undefined, digits: number): string {
  return x === null || x === undefined ? "n/a" : x.toFixed(digits);
}

/**
 * Render a stored timestamp in the LOCAL timezone as "YYYY-MM-DD HH:MM".
 *
 * Records store `ts` as UTC (`new Date().toISOString()`) and that stays the
 * on-disk format — it's unambiguous and comparable across machines/DST. Only
 * the display is localized. Previously this column was a raw `ts.slice(0,16)`,
 * which printed UTC and so read ~4h ahead of the wall clock on this box (EDT):
 * a 22:56 run showed as `02:56` the FOLLOWING day, which made runs hard to line
 * up against local-time logs like ~/scripts/vllm-startups.csv.
 *
 * getFullYear/getMonth/... (not the getUTC* variants) are what do the
 * conversion; toISOString() here would silently undo the whole point.
 */
function formatLocalTs(ts: string | null | undefined): string {
  if (!ts) return "-";
  const d = new Date(ts);
  // Malformed/legacy value: fall back to the old raw-prefix behaviour rather
  // than printing "Invalid Date".
  if (Number.isNaN(d.getTime())) return ts.slice(0, 16).replace("T", " ");
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}`
  );
}

function report(): void {
  const records = loadRecords();
  if (!records.length) {
    console.log(`No saved results in ${RESULTS_FILE}`);
    return;
  }
  const rows = records.map((r) => ({
    when: formatLocalTs(r.ts),
    tag: r.tag ?? r.serverVersion ?? "-",
    model: r.model.length > 34 ? r.model.slice(0, 33) + "…" : r.model,
    mean: fmt(r.aggregate?.decodeTpsMean, 1),
    med: fmt(r.aggregate?.decodeTpsMedian, 1),
    min: fmt(r.aggregate?.decodeTpsMin, 1),
    max: fmt(r.aggregate?.decodeTpsMax, 1),
    ttft: fmt(r.aggregate?.ttftMedian, 3),
    acc: r.spec ? `${r.spec.acceptLen.toFixed(2)}x` : "-",
  }));

  const cols: [string, keyof (typeof rows)[0], "l" | "r"][] = [
    ["when (local)", "when", "l"],
    ["tag", "tag", "l"],
    ["model", "model", "l"],
    ["tok/s mean", "mean", "r"],
    ["med", "med", "r"],
    ["min", "min", "r"],
    ["max", "max", "r"],
    ["TTFT med", "ttft", "r"],
    ["accept", "acc", "r"],
  ];
  const width = (head: string, key: keyof (typeof rows)[0]) =>
    Math.max(head.length, ...rows.map((r) => r[key].length));
  const widths = cols.map(([head, key]) => width(head, key));
  const line = (cells: string[]) =>
    cells.map((c, i) => (cols[i][2] === "l" ? c.padEnd(widths[i]) : c.padStart(widths[i]))).join("  ").trimEnd();

  console.log(`${records.length} run(s) from ${RESULTS_FILE}\n`);
  console.log(line(cols.map(([h]) => h)));
  console.log("-".repeat(widths.reduce((a, b) => a + b, 0) + 2 * (widths.length - 1)));
  for (const r of rows) console.log(line(cols.map(([, key]) => r[key])));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.report) {
    report();
    return;
  }
  const model = args.model ?? (await detectModel(args.host));
  const serverVersion = await detectServerVersion(args.host);
  const versionNote = serverVersion ? `  vLLM ${serverVersion}` : "";
  console.log(`Benchmarking  ${model}  @ ${args.host}  (max_tokens=${args.maxTokens})${versionNote}\n`);

  // Warm up before the spec-metrics baseline, so the throwaway's draft tokens
  // are not counted in the acceptance rate reported at the end.
  process.stdout.write("warming up ... ");
  let warmedUp = false;
  try {
    const w = await runPrompt(args.host, model, WARMUP_PROMPT, WARMUP_TOKENS);
    warmedUp = true;
    console.log(`${w.ttft === null ? "n/a" : w.ttft.toFixed(3)}s TTFT (discarded)\n`);
  } catch (e) {
    // A failed warmup is not a failed benchmark; the real prompts report their
    // own errors. Say so and carry on, with the first row's tax un-paid.
    console.log(`failed: ${(e as Error).message} — first row may read high\n`);
  }

  const specBefore = await scrapeSpecMetrics(args.host);

  const results: Result[] = [];
  const header =
    "prompt".padEnd(15) + pad("TTFT(s)", 9) + pad("tokens", 8) +
    pad("decode(s)", 10) + pad("decode tok/s", 14) + pad("total tok/s", 13);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const [label, prompt] of PROMPTS) {
    try {
      const r = await runPrompt(args.host, model, prompt, args.maxTokens);
      r.label = label;
      results.push(r);
      const decodeTime = r.ttft === null ? 0 : r.total - r.ttft;
      // A reasoning model that hits the cap mid-thought answers nothing; the
      // timings are still valid, but the row is not measuring a full reply.
      const note = !r.sawContent ? "  (thinking only)" : r.finishReason === "length" ? "  (truncated)" : "";
      console.log(
        label.padEnd(15) + pad(r.ttft === null ? "n/a" : r.ttft.toFixed(3), 9) +
        pad(r.completionTokens, 8) + pad(decodeTime.toFixed(3), 10) +
        pad(r.decodeTps > 0 ? r.decodeTps.toFixed(1) : "n/a", 14) +
        pad(r.totalTps.toFixed(1), 13) + note,
      );
    } catch (e) {
      console.log(`${label.padEnd(15)}  ERROR: ${(e as Error).message}`);
    }
  }

  const specAfter = await scrapeSpecMetrics(args.host);

  const agg = summarize(results);
  if (results.length) {
    console.log("-".repeat(header.length));
    console.log("\nAggregate:");
    if (agg.decodeTpsMean !== null) {
      console.log(
        `  decode tok/s   mean ${fmt(agg.decodeTpsMean, 1)}   median ${fmt(agg.decodeTpsMedian, 1)}` +
        `   min ${fmt(agg.decodeTpsMin, 1)}   max ${fmt(agg.decodeTpsMax, 1)}`,
      );
    }
    if (agg.ttftMean !== null) {
      console.log(`  TTFT (s)       mean ${fmt(agg.ttftMean, 3)}   median ${fmt(agg.ttftMedian, 3)}`);
    }
    console.log(`  total completion tokens: ${agg.completionTokens}`);
  }

  let spec: SpecSummary | null = null;
  if (specBefore && specAfter) {
    const dDrafts = specAfter.drafts - specBefore.drafts;
    const dDraftTok = specAfter.draftTokens - specBefore.draftTokens;
    const dAccepted = specAfter.accepted - specBefore.accepted;
    console.log("\nSpeculative decoding (MTP) — this run:");
    if (dDraftTok > 0) {
      const acceptRate = (100 * dAccepted) / dDraftTok;
      // accept length = target's guaranteed token + accepted draft tokens per step
      const acceptLen = dDrafts > 0 ? 1 + dAccepted / dDrafts : 0;
      spec = { drafts: dDrafts, draftTokens: dDraftTok, accepted: dAccepted, acceptRate, acceptLen };
      console.log(`  draft tokens proposed : ${Math.round(dDraftTok)}`);
      console.log(`  draft tokens accepted : ${Math.round(dAccepted)}`);
      console.log(`  acceptance rate       : ${acceptRate.toFixed(1)}%`);
      console.log(`  mean accept length    : ${acceptLen.toFixed(2)}  (~${acceptLen.toFixed(2)}x decode vs no spec)`);
    } else {
      console.log("  (no draft tokens recorded this run)");
    }
  } else if (specAfter === null) {
    console.log("\n(no speculative-decode metrics exposed — server likely running without MTP)");
  }

  // Persist last: a failed append should not cost the numbers already printed.
  if (args.save && results.length) {
    const rec: BenchRecord = {
      ts: new Date().toISOString(),
      tag: args.tag,
      host: args.host,
      model,
      serverVersion,
      maxTokens: args.maxTokens,
      warmup: warmedUp,
      results,
      aggregate: agg,
      spec,
    };
    try {
      saveRecord(rec);
      console.log(`\nAppended to ${RESULTS_FILE}  (--report to compare runs)`);
    } catch (e) {
      console.error(`\nWARNING: could not save results: ${(e as Error).message}`);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
