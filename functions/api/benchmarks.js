import {validateTable} from "../../landing/benchmarks.mjs";

const repoPattern = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/;
const headers = {"Content-Type": "application/json", "Cache-Control": "private, no-store",
  "X-Content-Type-Options": "nosniff"};

export async function benchmarkResponse(env, fetcher = fetch) {
  const repository = env.EGOLEARN_RESULTS_REPO;
  const token = env.EGOLEARN_RESULTS_TOKEN;
  if (!repoPattern.test(repository) || !token) {
    return new Response(JSON.stringify({error: "Benchmark source is not configured"}), {status: 503, headers});
  }
  let failure = "upstream request";
  try {
    const response = await fetcher(`https://huggingface.co/datasets/${repository}/resolve/main/benchmarks.json`, {
      headers: {Authorization: `Bearer ${token}`, "Cache-Control": "no-cache"},
      // Older Pages compatibility dates do not support Request.cache.
      cf: {cacheTtlByStatus: {"100-599": -1}}, signal: AbortSignal.timeout(10000),
    });
    failure = `upstream HTTP ${response.status}`;
    if (!response.ok || !response.body) throw new Error("Hub unavailable");
    failure = "feed read or size limit";
    // Bound the response even when the upstream omits Content-Length.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let size = 0;
    let text = "";
    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 4 * 1024 * 1024) throw new Error("Feed too large");
        text += decoder.decode(value, {stream: true});
      }
      text += decoder.decode();
    } finally {
      await reader.cancel();
    }
    failure = "feed validation";
    const data = validateTable(JSON.parse(text));
    if (data.results_repo !== repository) throw new Error("Unexpected results repository");
    const revision = response.headers.get("x-repo-commit");
    if (revision && /^[a-f0-9]{40}$/.test(revision)) data.revision = revision;
    return new Response(JSON.stringify(data), {headers});
  } catch {
    console.error(`Benchmark source unavailable: ${failure}`);
    return new Response(JSON.stringify({error: "Benchmark source is unavailable"}), {status: 502, headers});
  }
}

export function onRequestGet({env}) {
  return benchmarkResponse(env);
}
