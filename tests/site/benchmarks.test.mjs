import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {parseHTML} from "linkedom";
import {benchmarkResponse} from "../../functions/api/benchmarks.js";
import {formatMetric, startBenchmarks, validateTable} from "../../landing/benchmarks.mjs";

function fixture() {
  return {
    schema_version: 1, updated_at: "2026-01-01T00:00:00Z", results_repo: "test/results",
    columns: [{id: "accuracy", benchmark: "toy", metric: "accuracy", direction: "max",
      label: "Accuracy", description: "Synthetic accuracy", unit: "fraction"}],
    rows: [{id: "toy", condition: {id: "toy", label: "Toy", dataset: "Synthetic", regime: "offline",
      architecture: "ToyCNN", description: "Synthetic tests"}, expected_seeds: [1],
      cells: {accuracy: {mean: 0.4328, sd: null, n: 1, expected_n: 1, status: "complete",
        protocols: [{sha256: "a".repeat(64), version: "1", settings: {}, input_sha256: {}, input_scope: "Synthetic",
          evaluation_seed: 0, code_commit: null}]}},
      runs: [{id: "a".repeat(64), seed: 1, checkpoint: {repo_id: "test/model", revision: "b".repeat(40),
        sha256: "c".repeat(64), weights_sha256: "d".repeat(64), metadata_sha256: "e".repeat(64),
        step: 12, private: true}, provenance: {training_config_sha256: "f".repeat(64),
        training_code_commit: null, legacy_evaluation: false}}]}],
  };
}

test("metrics display percentages, unavailable cells and singleton uncertainty correctly", () => {
  const data = validateTable(fixture());
  const column = data.columns[0];
  const cell = data.rows[0].cells.accuracy;
  assert.deepEqual(formatMetric(cell, column), {value: "43.3", sd: ""});
  assert.deepEqual(formatMetric({...cell, status: "incomplete", mean: null}, column), {value: "—", sd: ""});
  assert.deepEqual(formatMetric({...cell, sd: 0.014}, column), {value: "43.3", sd: "±1.4"});
});

test("the feed rejects malformed, incomplete and nonfinite aggregates", () => {
  for (const mutate of [
    data => { data.schema_version = 999; },
    data => { data.local_config = {private_path: "/private/run"}; },
    data => { data.rows[0].cells.accuracy.mean = NaN; },
    data => { data.rows[0].cells.accuracy.sd = 0; },
    data => { data.rows[0].cells.accuracy.status = "incomplete"; },
    data => { data.rows[0].cells.accuracy.n = 0; },
    data => { data.rows[0].runs[0].checkpoint.repo_id = "javascript:alert(1)"; },
    data => { data.rows[0].runs[0].checkpoint.revision = "main"; },
  ]) {
    const data = fixture();
    mutate(data);
    assert.throws(() => validateTable(data));
  }
});

test("the server forwards credentials only to the configured Hub source and disables shared caching", async () => {
  const token = "synthetic-secret";
  const response = await benchmarkResponse({EGOLEARN_RESULTS_REPO: "test/results", EGOLEARN_RESULTS_TOKEN: token}, async (url, options) => {
    assert.equal(url, "https://huggingface.co/datasets/test/results/resolve/main/benchmarks.json");
    assert.equal(options.headers.Authorization, `Bearer ${token}`);
    assert.equal(options.cache, "no-store");
    return new Response(JSON.stringify(fixture()), {headers: {"x-repo-commit": "d".repeat(40)}});
  });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "private, no-store");
  const body = await response.text();
  assert.ok(!body.includes(token));
  assert.equal(JSON.parse(body).revision, "d".repeat(40));
});

test("the server fails closed for missing configuration, upstream failures and mismatched feeds", async () => {
  assert.equal((await benchmarkResponse({}, () => assert.fail("must not fetch"))).status, 503);
  const env = {EGOLEARN_RESULTS_REPO: "test/results", EGOLEARN_RESULTS_TOKEN: "synthetic-secret"};
  for (const upstream of [
    new Response("denied", {status: 401}),
    new Response("<html>login</html>"),
    new Response(JSON.stringify({...fixture(), results_repo: "other/results"})),
    new Response("x".repeat(4 * 1024 * 1024 + 1)),
  ]) {
    const response = await benchmarkResponse(env, async () => upstream);
    assert.equal(response.status, 502);
    assert.ok(!(await response.text()).includes(env.EGOLEARN_RESULTS_TOKEN));
  }
});

test("a newly published row is available on the next fetch without rebuilding", async () => {
  const data = fixture();
  const env = {EGOLEARN_RESULTS_REPO: "test/results", EGOLEARN_RESULTS_TOKEN: "synthetic-secret"};
  const fetcher = async () => new Response(JSON.stringify(data));
  assert.equal((await (await benchmarkResponse(env, fetcher)).json()).rows.length, 1);
  const second = structuredClone(data.rows[0]);
  second.id = "second";
  second.condition.label = "Second model";
  second.runs[0].id = "e".repeat(64);
  second.runs[0].checkpoint.sha256 = "f".repeat(64);
  data.rows.push(second);
  assert.equal((await (await benchmarkResponse(env, fetcher)).json()).rows.length, 2);
});

test("the landing page contains the dynamic table and no hand-entered result row", () => {
  const html = readFileSync(new URL("../../landing/index.html", import.meta.url), "utf8");
  assert.match(html, /id="benchmark-table"/);
  assert.match(html, /src="benchmarks.mjs"/);
  assert.doesNotMatch(html, /43\.3<span/);
});

test("rendering keeps the last successful table during failures and safely adds new rows", async () => {
  const {document} = parseHTML('<html><body><div id="table"></div><p id="status"></p></body></html>');
  globalThis.document = document;
  const container = document.getElementById("table");
  const status = document.getElementById("status");
  let data = fixture();
  let available = true;
  const fetcher = async () => available ? new Response(JSON.stringify(data)) : new Response("{}", {status: 502});
  const controller = startBenchmarks({container, status, fetcher, interval: 1000000});
  try {
    await new Promise(resolve => setImmediate(resolve));
    assert.match(container.textContent, /43\.3/);
    assert.equal(container.querySelectorAll(".meth").length, 1);
    const original = container.innerHTML;
    available = false;
    await controller.refresh();
    assert.equal(container.innerHTML, original);
    assert.match(status.textContent, /Live refresh is unavailable/);
    available = true;
    data = {...fixture(), schema_version: 99};
    await controller.refresh();
    assert.equal(container.innerHTML, original);
    data = fixture();
    const row = structuredClone(data.rows[0]);
    row.id = "second";
    row.condition.label = '<img src=x onerror="alert(1)">';
    data.rows.push(row);
    await controller.refresh();
    assert.equal(container.querySelectorAll(".meth").length, 2);
    assert.ok(container.textContent.includes(row.condition.label));
    assert.equal(container.querySelectorAll("img").length, 0);
    assert.equal(container.querySelectorAll("details a").length, 4);
    assert.match(status.textContent, /Refreshes every minute/);
  } finally {
    controller.stop();
    delete globalThis.document;
  }
});
