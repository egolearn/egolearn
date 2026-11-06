const repoPattern = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/;
const digestPattern = /^[a-f0-9]{64}$/;
const commitPattern = /^[a-f0-9]{40}$/;

function fields(value, required, optional = []) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      required.some(key => !Object.hasOwn(value, key)) ||
      Object.keys(value).some(key => !required.includes(key) && !optional.includes(key))) {
    throw new Error("Unexpected benchmark fields");
  }
}

export function validateTable(data) {
  fields(data, ["schema_version", "updated_at", "results_repo", "columns", "rows"], ["revision"]);
  if (!data || ![1, 2].includes(data.schema_version) || !repoPattern.test(data.results_repo) ||
      !Number.isFinite(Date.parse(data.updated_at)) || !Array.isArray(data.columns) ||
      !Array.isArray(data.rows) || data.columns.length > 100 || data.rows.length > 1000) {
    throw new Error("Invalid benchmark feed");
  }
  const columns = new Set();
  for (const column of data.columns) {
    fields(column, ["id", "benchmark", "metric", "label", "description", "unit", "direction"]);
    if (!column || typeof column.id !== "string" || columns.has(column.id) ||
        typeof column.label !== "string" || typeof column.description !== "string" ||
        !["fraction", "number"].includes(column.unit)) throw new Error("Invalid column");
    columns.add(column.id);
  }
  const rows = new Set();
  for (const row of data.rows) {
    fields(row, ["id", "condition", "expected_seeds", "cells", "runs"]);
    fields(row.condition, ["id", "label", "dataset", "regime", "architecture", "description"]);
    const reference = data.schema_version === 2 && Array.isArray(row.expected_seeds) &&
      row.expected_seeds.length === 1 && row.expected_seeds[0] === null;
    if (!row || typeof row.id !== "string" || rows.has(row.id) || !row.condition ||
        !["label", "dataset", "regime", "architecture", "description"].every(k => typeof row.condition[k] === "string") ||
        !Array.isArray(row.expected_seeds) || !row.expected_seeds.length ||
        (!reference && !row.expected_seeds.every(Number.isSafeInteger)) ||
        new Set(row.expected_seeds).size !== row.expected_seeds.length ||
        !Array.isArray(row.runs) || !row.runs.length || !row.cells) throw new Error("Invalid row");
    rows.add(row.id);
    const seeds = new Set();
    const checkpoints = new Set();
    fields(row.cells, [...columns]);
    for (const run of row.runs) {
      fields(run, ["id", "seed", "checkpoint", "provenance"]);
      fields(run.provenance, ["training_config_sha256", "training_code_commit", "legacy_evaluation"],
        data.schema_version === 2 ? ["reference"] : []);
      if (Object.hasOwn(run.provenance, "reference") && typeof run.provenance.reference !== "boolean") {
        throw new Error("Invalid reference provenance");
      }
      if (reference ? run.provenance.reference !== true || run.provenance.training_config_sha256 !== null
        : run.provenance.reference === true || !digestPattern.test(run.provenance.training_config_sha256)) {
        throw new Error("Invalid training provenance");
      }
      const checkpoint = run.checkpoint;
      fields(checkpoint, ["repo_id", "revision", "sha256", "weights_sha256", "metadata_sha256", "step", "private"]);
      const local = checkpoint.repo_id === null;
      const validLocation = local
        ? checkpoint.revision === null && checkpoint.private === null
        : repoPattern.test(checkpoint.repo_id) && commitPattern.test(checkpoint.revision) && typeof checkpoint.private === "boolean";
      if (!digestPattern.test(run.id) || !row.expected_seeds.includes(run.seed) || seeds.has(run.seed) ||
          !validLocation || !digestPattern.test(checkpoint.sha256) || checkpoints.has(checkpoint.sha256)) {
        throw new Error("Invalid run reference");
      }
      seeds.add(run.seed);
      checkpoints.add(checkpoint.sha256);
    }
    for (const column of data.columns) {
      const cell = row.cells[column.id];
      const notApplicable = data.schema_version === 2 && cell?.status === "not_applicable";
      fields(cell, ["mean", "sd", "n", "expected_n", "status", "protocols", ...(notApplicable ? ["reason"] : [])]);
      if (!cell || (!["complete", "incomplete", "incompatible"].includes(cell.status) && !notApplicable) ||
          !Number.isSafeInteger(cell.n) || cell.n < 0 || cell.n > row.runs.length ||
          cell.expected_n !== row.expected_seeds.length || !Array.isArray(cell.protocols)) {
        throw new Error("Invalid metric");
      }
      if (notApplicable && (cell.n !== 0 || cell.protocols.length !== 0 ||
          typeof cell.reason !== "string" || !cell.reason.trim())) {
        throw new Error("Invalid benchmark applicability");
      }
      for (const protocol of cell.protocols) {
        fields(protocol, ["sha256", "version", "settings", "input_sha256", "input_scope", "evaluation_seed", "code_commit"]);
        if (!digestPattern.test(protocol.sha256) || typeof protocol.version !== "string" ||
            !protocol.settings || !protocol.input_sha256 || typeof protocol.input_scope !== "string") {
          throw new Error("Invalid protocol");
        }
      }
      if (cell.status === "complete") {
        if (cell.n !== cell.expected_n || !Number.isFinite(cell.mean) ||
            cell.protocols.length !== 1 ||
            (column.unit === "fraction" && (cell.mean < 0 || cell.mean > 1)) ||
            (cell.n === 1 ? cell.sd !== null : !Number.isFinite(cell.sd) || cell.sd < 0)) {
          throw new Error("Invalid aggregate");
        }
      } else if (cell.mean !== null || cell.sd !== null) throw new Error("Incomplete aggregate");
    }
  }
  return data;
}

export function formatMetric(cell, column) {
  if (cell.status === "not_applicable") return {value: "N/A", sd: ""};
  if (cell.status !== "complete") return {value: "—", sd: ""};
  const factor = column.unit === "fraction" ? 100 : 1;
  return {value: (cell.mean * factor).toFixed(1),
    sd: cell.sd === null ? "" : `±${(cell.sd * factor).toFixed(1)}`};
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function link(text, url) {
  const node = element("a", text);
  node.href = url;
  return node;
}

export function renderTable(data, container) {
  validateTable(data);
  const table = element("table", undefined, "bench");
  table.setAttribute("aria-describedby", "benchmark-note");
  const caption = element("caption", "EgoLearn benchmark results", "visually-hidden");
  table.append(caption);
  const head = element("thead");
  const headings = element("tr");
  for (const label of ["method", ...data.columns.map(column => `${column.label}${column.unit === "fraction" ? " (%)" : ""}`)]) {
    const heading = element("th", label);
    heading.scope = "col";
    headings.append(heading);
  }
  head.append(headings);
  table.append(head);
  for (const dataset of [...new Set(data.rows.map(row => row.condition.dataset))]) {
    const body = element("tbody");
    const group = element("tr", undefined, "group");
    const heading = element("th", `trained on ${dataset}`);
    heading.colSpan = data.columns.length + 1;
    heading.scope = "rowgroup";
    group.append(heading);
    body.append(group);
    for (const row of data.rows.filter(row => row.condition.dataset === dataset)) {
      const tr = element("tr");
      const method = element("td");
      method.append(element("span", row.condition.label, "meth"));
      method.append(element("span", `${row.condition.regime} · ${row.condition.architecture}`, "kind"));
      const details = element("details", undefined, "benchmark-details");
      const reference = row.expected_seeds[0] === null;
      details.append(element("summary", reference ? "1 reference checkpoint · sources"
        : `${row.runs.length} of ${row.expected_seeds.length} seeds · sources`));
      details.append(element("p", row.condition.description));
      if (row.runs.some(run => run.provenance?.legacy_evaluation || !run.provenance?.training_code_commit)) {
        details.append(element("p", "Some source revisions were not recorded. See each evaluation record for provenance limits."));
      }
      for (const run of row.runs) {
        const p = element("p");
        const identity = run.seed === null ? "Training seed unknown" : `Seed ${run.seed}`;
        const record = `https://huggingface.co/datasets/${data.results_repo}/blob/${data.revision || "main"}/runs/${run.id}.json`;
        if (run.checkpoint.repo_id === null) {
          p.append(document.createTextNode(`${identity} · checkpoint not uploaded`));
        } else {
          const model = `https://huggingface.co/${run.checkpoint.repo_id}/tree/${run.checkpoint.revision}`;
          p.append(link(`${identity} · weights${run.checkpoint.private ? " (restricted)" : ""}`, model));
        }
        p.append(document.createTextNode(" · "), link("evaluation record", record));
        details.append(p);
      }
      method.append(details);
      tr.append(method);
      for (const column of data.columns) {
        const cell = row.cells[column.id];
        const formatted = formatMetric(cell, column);
        const td = element("td", formatted.value);
        if (formatted.sd) td.append(element("span", formatted.sd, "sd"));
        const sizes = [...new Set(cell.protocols.map(p => p.settings.size).filter(Number.isFinite))];
        const resolution = sizes.length ? ` Evaluation resolution: ${sizes.join(", ")} px.` : "";
        td.title = cell.status === "not_applicable" ? cell.reason
          : `${column.description} ${cell.n}/${cell.expected_n} ${reference ? "checkpoints" : "seeds"}; ${cell.status}.${resolution}`;
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
  }
  const fragment = document.createDocumentFragment();
  fragment.append(table);
  if (!data.rows.length) fragment.append(element("p", "No evaluated runs have been published yet."));
  container.replaceChildren(fragment);
}

export function startBenchmarks({container, status, fetcher = fetch, interval = 60000}) {
  let busy = false;
  let lastUpdated = null;
  let lastData = null;
  async function refresh() {
    if (busy || document.hidden) return;
    busy = true;
    try {
      const response = await fetcher("./api/benchmarks", {cache: "no-store", signal: AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error("Results unavailable");
      const data = validateTable(await response.json());
      if (data.revision !== undefined && !commitPattern.test(data.revision)) throw new Error("Invalid feed revision");
      const serialized = JSON.stringify(data);
      if (serialized !== lastData) {
        renderTable(data, container);
        lastData = serialized;
        lastUpdated = new Date(data.updated_at).toLocaleString();
      }
      const message = `Results updated ${lastUpdated}. Refreshes every minute.`;
      if (status.textContent !== message) status.textContent = message;
    } catch {
      const message = lastUpdated
        ? `Showing results from ${lastUpdated}. Live refresh is unavailable; retrying every minute.`
        : "Published results are temporarily unavailable. Retrying every minute.";
      if (status.textContent !== message) status.textContent = message;
    } finally {
      busy = false;
    }
  }
  const timer = setInterval(refresh, interval);
  document.addEventListener("visibilitychange", refresh);
  refresh();
  return {refresh, stop() {clearInterval(timer); document.removeEventListener("visibilitychange", refresh);}};
}

if (typeof document !== "undefined") {
  const container = document.getElementById("benchmark-table");
  const status = document.getElementById("benchmark-status");
  if (container && status) startBenchmarks({container, status});
}
