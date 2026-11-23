import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {parseHTML} from "linkedom";
import {fetchBenchmarks, formatMetric, renderTable, startBenchmarks, validateTable} from "../../landing/benchmarks.mjs";

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

test("the Python-generated schema-1 feed validates with all nine default metrics", () => {
  const historical = JSON.parse(readFileSync(new URL("../fixtures/publication-v1.json", import.meta.url), "utf8"));
  const data = validateTable(historical.table);
  assert.equal(data.columns.length, 9);
  assert.equal(data.rows[0].runs.length, 3);
  for (const column of data.columns) {
    assert.deepEqual(formatMetric(data.rows[0].cells[column.id], column), {value: "20.0", sd: "±10.0"});
  }
});

test("mixed schema feeds separate modalities and render unknown-seed references", () => {
  const mixed = JSON.parse(readFileSync(new URL("../fixtures/publication-v2.json", import.meta.url), "utf8"));
  const data = validateTable(mixed.table);
  const reference = data.rows.find(row => row.expected_seeds[0] === null);
  assert.equal(reference.runs[0].seed, null);
  assert.equal(reference.cells.labeled_s_linear.n, 1);
  assert.equal(reference.cells.labeled_s_linear.sd, null);
  assert.equal(reference.cells.labeled_s_4afc.status, "not_applicable");
  assert.deepEqual(formatMetric(reference.cells.labeled_s_4afc, data.columns[0]), {value: "N/A", sd: ""});
  const {document} = parseHTML('<html><body><div id="table"></div></body></html>');
  globalThis.document = document;
  try {
    const container = document.getElementById("table");
    renderTable(data, container);
    assert.equal(container.querySelectorAll("details, summary").length, 0);
    assert.doesNotMatch(container.textContent, /sources|Training seed unknown|Seed null/);
    const multimodal = container.querySelector('table[data-modality="multimodal"]');
    const visual = container.querySelector('table[data-modality="visual-only"]');
    assert.equal(container.querySelectorAll("table").length, 2);
    assert.equal(multimodal.querySelectorAll("thead th").length, 3);
    assert.equal(visual.querySelectorAll("thead th").length, 2);
    assert.equal(multimodal.querySelector(".meth").textContent, "Toy");
    assert.equal(visual.querySelector(".meth").textContent, reference.condition.label);
    assert.doesNotMatch(container.textContent, /N\/A/);
    const cells = [...container.querySelectorAll("td")];
    assert.ok(cells.some(cell => cell.title.includes("112 px")));
  } finally {
    delete globalThis.document;
  }
  for (const mutate of [
    row => { row.expected_seeds = [null, 1]; },
    row => { row.runs[0].provenance.reference = false; },
    row => { row.runs[0].provenance.training_config_sha256 = "a".repeat(64); },
    row => { row.cells.labeled_s_4afc.reason = " "; },
    row => { row.cells.labeled_s_4afc.n = 1; },
    row => { row.cells.labeled_s_4afc.mean = 0.5; },
  ]) {
    const changed = structuredClone(data);
    mutate(changed.rows.find(row => row.expected_seeds[0] === null));
    assert.throws(() => validateTable(changed));
  }
});

test("visual-only tables omit image-text benchmarks, including ones published after the run", () => {
  const konkle = cells => structuredClone(cells.labeled_s_4afc);
  const render = data => {
    const {document} = parseHTML('<html><body><div id="table"></div></body></html>');
    globalThis.document = document;
    try {
      const container = document.getElementById("table");
      renderTable(data, container);
      return Object.fromEntries([...container.querySelectorAll("table")].map(table => [table.dataset.modality, {
        rows: [...table.querySelectorAll("tbody tr")].map(row => row.dataset.resultId).sort(),
        headings: [...table.querySelectorAll("thead th")].map(cell => cell.textContent),
      }]));
    } finally {
      delete globalThis.document;
    }
  };
  const load = () => JSON.parse(readFileSync(new URL("../fixtures/publication-v2.json", import.meta.url), "utf8")).table;

  // Konkle 4AFC was published after the visual-only reference, so the feed records it
  // as missing for that run. A text model with one inapplicable image-text benchmark
  // remains multimodal.
  const data = load();
  data.columns.splice(1, 0, {...data.columns[0], id: "konkle_4afc", benchmark: "konkle_4afc",
    label: "Konkle 4AFC"});
  const visual = data.rows.find(row => row.id === "reference");
  const multimodal = data.rows.find(row => row.id === "toy");
  visual.cells.konkle_4afc = {mean: null, sd: null, n: 0, expected_n: 1, status: "incomplete", protocols: []};
  multimodal.cells.konkle_4afc = konkle(multimodal.cells);
  const partial = structuredClone(multimodal);
  partial.id = "partial-text";
  partial.cells.labeled_s_4afc = structuredClone(visual.cells.labeled_s_4afc);
  data.rows.push(partial);
  assert.deepEqual(render(data), {
    "multimodal": {rows: ["partial-text", "toy"],
      headings: ["method", "Labeled-S 4AFC (%)", "Konkle 4AFC (%)", "Labeled-S linear (%)"]},
    "visual-only": {rows: ["reference"], headings: ["method", "Labeled-S linear (%)"]},
  });

  // Konkle 4AFC as the only image-text benchmark still separates the modalities.
  const only = load();
  only.columns[0] = {...only.columns[0], id: "konkle_4afc", benchmark: "konkle_4afc", label: "Konkle 4AFC"};
  for (const row of only.rows) {
    row.cells.konkle_4afc = konkle(row.cells);
    delete row.cells.labeled_s_4afc;
  }
  assert.deepEqual(render(only), {
    "multimodal": {rows: ["toy"], headings: ["method", "Konkle 4AFC (%)", "Labeled-S linear (%)"]},
    "visual-only": {rows: ["reference"], headings: ["method", "Labeled-S linear (%)"]},
  });
});

test("dataset tables group rows and keep scores aligned with reordered columns", () => {
  const mixed = JSON.parse(readFileSync(new URL("../fixtures/publication-v2.json", import.meta.url), "utf8"));
  const data = mixed.table;
  const visual = data.rows.find(row => row.id === "reference");
  const multimodal = data.rows.find(row => row.id === "toy");
  visual.condition.label = "BabyCL (visual-only)";
  data.columns.splice(1, 0, {...data.columns[1], id: "shape_preference",
    benchmark: "shape_bias", label: "Shape preference"});
  // Column aliases must not change modality classification.
  data.columns[0].id = "word_matching";
  for (const row of data.rows) {
    row.condition.dataset = "BabyView";
    row.cells.word_matching = row.cells.labeled_s_4afc;
    delete row.cells.labeled_s_4afc;
    row.cells.shape_preference = {...structuredClone(visual.cells.labeled_s_linear), mean: 0.75};
  }
  for (const original of [...data.rows]) {
    const row = structuredClone(original);
    row.id += "-second-dataset";
    row.condition.dataset = "SAYCam-S";
    data.rows.push(row);
  }
  const incomplete = data.rows.find(row => row.id === `${multimodal.id}-second-dataset`);
  incomplete.cells.word_matching = {mean: null, sd: null, n: 0, expected_n: 1,
    status: "incomplete", protocols: []};
  const imagenet = structuredClone(visual);
  imagenet.id = "imagenet-supervised-reference";
  imagenet.condition.dataset = "ImageNet";
  imagenet.condition.label = "ImageNet supervised";
  data.rows.push(imagenet);
  const before = structuredClone(data);
  const {document} = parseHTML('<html><body><div id="table"></div></body></html>');
  globalThis.document = document;
  try {
    const container = document.getElementById("table");
    renderTable(data, container);
    assert.deepEqual(data, before);
    assert.equal(container.querySelectorAll(".result-dataset").length, 3);
    assert.deepEqual([...container.querySelectorAll(".result-dataset h3")].map(node => node.textContent),
      ["trained on SAYCam-S", "trained on BabyView", "Others"]);
    const rendered = [...container.querySelectorAll("tbody tr")];
    assert.deepEqual(rendered.map(row => row.dataset.resultId).sort(),
      data.rows.map(row => row.id).sort());
    for (const section of container.querySelectorAll(".result-dataset")) {
      const others = section.querySelector("h3").textContent === "Others";
      assert.equal(section.querySelectorAll("table").length, others ? 1 : 2);
      for (const table of section.querySelectorAll("table")) {
        assert.equal(table.querySelectorAll("tbody tr").length, 1);
        const isVisual = table.dataset.modality === "visual-only";
        const row = table.querySelector("tbody tr");
        const source = data.rows.find(source => source.id === row.dataset.resultId);
        assert.equal(section.querySelector("h3").textContent,
          source.condition.dataset === "ImageNet" ? "Others" : `trained on ${source.condition.dataset}`);
        assert.equal(row.querySelector(".meth").textContent, source.condition.label);
        assert.equal(table.querySelector("caption").textContent, isVisual ? "visual-only models" : "multimodal models");
        for (const id of table.getAttribute("aria-labelledby").split(" ")) {
          assert.ok(document.getElementById(id));
        }
        const columns = isVisual ? [data.columns[2], data.columns[1]]
          : [data.columns[0], data.columns[2], data.columns[1]];
        assert.deepEqual([...table.querySelectorAll("thead th")].map(cell => cell.textContent),
          ["method", ...columns.map(column => `${column.label} (%)`)]);
        const actual = [...row.querySelectorAll("td")].slice(1).map(cell => cell.textContent);
        const expected = columns.map(column => {
          const {value, sd} = formatMetric(source.cells[column.id], column);
          return value + sd;
        });
        assert.deepEqual(actual, expected);
        assert.equal(row.querySelectorAll("details, summary, a").length, 0);
      }
    }
    assert.equal(container.querySelectorAll("[id]").length,
      new Set([...container.querySelectorAll("[id]")].map(node => node.id)).size);
  } finally {
    delete globalThis.document;
  }
});

test("every published multimodal result appears under its published label", () => {
  const {table: data} = JSON.parse(readFileSync(new URL("../fixtures/publication-v2.json", import.meta.url), "utf8"));
  const current = data.rows.find(row => row.id === "toy");
  current.id = current.condition.id = "babycl-supcon-babyview";
  current.condition.label = "BabyCL";
  const visual = data.rows.find(row => row.id === "reference");
  for (const row of data.rows) row.condition.dataset = "BabyView";
  const second = structuredClone(current);
  second.id = second.condition.id = "babycl-babyview-legacy-offline-refined-seed1337";
  second.condition.label = "BabyCL (legacy offline-refined, 60,000 steps)";
  data.rows.unshift(second);
  const {document} = parseHTML('<html><body><div id="table"></div></body></html>');
  globalThis.document = document;
  try {
    const container = document.getElementById("table");
    renderTable(data, container);
    const rows = [...container.querySelectorAll('table[data-modality="multimodal"] tbody tr')];
    assert.deepEqual(rows.map(row => [row.dataset.resultId, row.querySelector(".meth").textContent]),
      [[second.id, second.condition.label], [current.id, current.condition.label]]);
    assert.equal(container.querySelector('table[data-modality="visual-only"] tbody tr').dataset.resultId, visual.id);
  } finally {
    delete globalThis.document;
  }
});

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

test("the browser reads the public Hub feed without credentials and records its revision", async () => {
  const data = await fetchBenchmarks("test/results", async (url, options) => {
    assert.equal(url, "https://huggingface.co/datasets/test/results/resolve/main/benchmarks.json");
    assert.equal(options.credentials, "omit");
    assert.equal(options.cache, "no-store");
    assert.equal(options.headers, undefined);
    assert.ok(options.signal instanceof AbortSignal);
    return new Response(JSON.stringify(fixture()), {headers: {"x-repo-commit": "d".repeat(40)}});
  });
  assert.equal(data.revision, "d".repeat(40));
});

test("the browser rejects invalid sources, upstream failures and mismatched feeds", async () => {
  await assert.rejects(fetchBenchmarks(undefined, () => assert.fail("must not fetch")));
  await assert.rejects(fetchBenchmarks("../private", () => assert.fail("must not fetch")));
  for (const upstream of [
    new Response("denied", {status: 401}),
    new Response("throttled", {status: 429}),
    new Response("<html>login</html>"),
    new Response(JSON.stringify({...fixture(), results_repo: "other/results"})),
    new Response(JSON.stringify(fixture()), {headers: {"x-repo-commit": "invalid"}}),
    new Response(JSON.stringify({...fixture(), revision: "invalid"})),
    new Response("x".repeat(4 * 1024 * 1024 + 1)),
  ]) {
    await assert.rejects(fetchBenchmarks("test/results", async () => upstream));
  }
  await assert.rejects(fetchBenchmarks("test/results", async () => { throw new Error("Network error"); }));
});

test("a newly published row is available on the next fetch without rebuilding", async () => {
  const data = fixture();
  const fetcher = async () => new Response(JSON.stringify(data));
  assert.equal((await fetchBenchmarks("test/results", fetcher)).rows.length, 1);
  const second = structuredClone(data.rows[0]);
  second.id = "second";
  second.condition.label = "Second model";
  second.runs[0].id = "e".repeat(64);
  second.runs[0].checkpoint.sha256 = "f".repeat(64);
  data.rows.push(second);
  assert.equal((await fetchBenchmarks("test/results", fetcher)).rows.length, 2);
});

test("the landing page contains the dynamic table and no hand-entered result row", () => {
  const html = readFileSync(new URL("../../landing/index.html", import.meta.url), "utf8");
  assert.match(html, /id="benchmark-table"[^>]*data-results-repo="egolearn\/egolearn-results"/);
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
  container.dataset.resultsRepo = "test/results";
  const controller = startBenchmarks({container, status, fetcher, interval: 1000000});
  try {
    await new Promise(resolve => setImmediate(resolve));
    assert.match(container.textContent, /43\.3/);
    assert.equal(container.querySelectorAll(".meth").length, 1);
    assert.equal(status.textContent, "");
    assert.equal(status.hidden, true);
    const table = container.querySelector("table");
    await controller.refresh();
    assert.equal(container.querySelector("table"), table);
    const original = container.innerHTML;
    available = false;
    await controller.refresh();
    assert.equal(container.innerHTML, original);
    assert.match(status.textContent, /Live refresh is unavailable/);
    assert.equal(status.hidden, false);
    const failureText = status.firstChild;
    await controller.refresh();
    assert.equal(status.firstChild, failureText);
    available = true;
    await controller.refresh();
    assert.equal(container.querySelector("table"), table);
    assert.equal(status.textContent, "");
    assert.equal(status.hidden, true);
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
    assert.equal(container.querySelectorAll("details, summary").length, 0);
    assert.equal(status.textContent, "");
    assert.equal(status.hidden, true);
  } finally {
    controller.stop();
    delete globalThis.document;
  }
});

test("results-only records display scores without source controls", () => {
  const {document} = parseHTML('<html><body><div id="table"></div></body></html>');
  globalThis.document = document;
  try {
    const data = fixture();
    const checkpoint = data.rows[0].runs[0].checkpoint;
    checkpoint.repo_id = null;
    checkpoint.revision = null;
    checkpoint.private = null;
    const container = document.getElementById("table");
    renderTable(data, container);
    assert.match(container.textContent, /43\.3/);
    assert.equal(container.querySelectorAll("details, summary, a").length, 0);
    checkpoint.revision = "b".repeat(40);
    assert.throws(() => validateTable(data));
  } finally {
    delete globalThis.document;
  }
});
