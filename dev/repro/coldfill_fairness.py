"""How much does one long cold prompt slow everyone else?

4 streaming replies reach steady decode, then a fresh ~N-token prompt (unique,
so no prefix-cache hit) with a needle arrives. Measures each stream's decode
rate before / during / after the read-in, the read-in time (TTFT of the long
request), and that the needle answer is right. Waits for an idle engine first
and records other traffic seen during the run.

  mixfill.py <label> [long_tokens=100000] [out.jsonl]
"""

import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

URL = os.environ.get("GLM_URL", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ("river stone lantern harbor meadow copper falcon willow ember signal orchard canyon "
         "glacier thunder velvet saffron compass mirror garden quarry beacon timber marble "
         "harvest station circuit lattice anchor prairie summit ledger cobalt dynamo").split()


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=5) as r:
        t = r.read().decode()
    def g(name):
        m = re.search(r"^%s\{[^}]*\} ([0-9.e+]+)" % re.escape(name), t, re.M)
        return float(m.group(1)) if m else 0.0
    return g("vllm:num_requests_running"), g("vllm:num_requests_waiting")


def wait_idle(hold=8, limit=900):
    t0, quiet = time.time(), None
    while time.time() - t0 < limit:
        run, wait = metrics()
        if run == 0 and wait == 0:
            quiet = quiet or time.time()
            if time.time() - quiet >= hold:
                return time.time() - t0
        else:
            quiet = None
        time.sleep(1)
    return None


def stream(messages, max_tokens, log, label):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Bench": "coldfill_fairness"})
    rec = {"label": label, "sent": time.time(), "chunks": [], "text": "", "reason": "", "usage": None}
    log.append(rec)
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                rec["usage"] = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
                if piece:
                    rec["chunks"].append((time.time(), len(piece)))
                    rec["text"] += delta.get("content") or ""
                    rec["reason"] += (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
    rec["done"] = time.time()


def rate(rec, a, b):
    """Tokens/s of a stream inside [a, b], scaling chars to its completion_tokens."""
    chars = sum(n for t, n in rec["chunks"])
    toks = (rec["usage"] or {}).get("completion_tokens") or 0
    if not chars or not toks or b <= a:
        return None
    inside = sum(n for t, n in rec["chunks"] if a <= t <= b)
    return inside * toks / chars / (b - a)


def main():
    label = sys.argv[1]
    n_long = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    out = sys.argv[3] if len(sys.argv) > 3 else "mixfill.jsonl"
    waited = wait_idle()
    rnd = random.Random()
    nonce = "%x" % rnd.getrandbits(64)
    code = rnd.randint(100000, 999999)
    words = [rnd.choice(WORDS) for _ in range(int(n_long / 1.25))]
    words.insert(int(len(words) * 0.6), "The secret code for run %s is %d." % (nonce, code))
    long_msgs = [{"role": "user", "content": "Run %s. Archive follows.\n\n%s\n\nWhat is the secret code for run %s? "
                  "Reply with only the number." % (nonce, " ".join(words), nonce)}]
    topics = ["the history of lighthouses", "how glaciers shape valleys", "the craft of bookbinding",
              "why rivers meander"]
    log, others = [], []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                run, wait = metrics()
                others.append((time.time(), run, wait))
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    th = [threading.Thread(target=stream, args=([{"role": "user", "content": "(%s-%d) Write a detailed, "
          "flowing 900-word essay on %s. No headings." % (nonce, i, t)}], 1400, log, "s%d" % i))
          for i, t in enumerate(topics)]
    for t in th:
        t.start()
    time.sleep(12)                                  # let the four reach steady decode
    lt = threading.Thread(target=stream, args=(long_msgs, 300, log, "long"))
    t_send = time.time()
    lt.start()
    lt.join()
    for t in th:
        t.join()
    stop.set()
    L = [r for r in log if r["label"] == "long"][0]
    t_first = L["chunks"][0][0] if L["chunks"] else L["done"]
    fill = t_first - t_send
    S = [r for r in log if r["label"] != "long"]
    res = {"label": label, "idle_wait_s": waited, "long_prompt_tokens": (L["usage"] or {}).get("prompt_tokens"),
           "fill_s": round(fill, 2), "needle_ok": str(code) in (L["text"] + L["reason"]), "needle_answer": L["text"].strip()[:20]}
    if res["long_prompt_tokens"]:
        res["fill_tok_per_s"] = round(res["long_prompt_tokens"] / fill)
    before, during, after = [], [], []
    for r in S:
        f0 = r["chunks"][0][0] if r["chunks"] else None
        b = rate(r, max(f0 + 1, t_send - 8), t_send) if f0 else None
        d = rate(r, t_send + 1, t_first)
        a = rate(r, t_first + 1, min(t_first + 9, r.get("done", t_first)))
        before.append(b); during.append(d); after.append(a)
    def avg(x):
        x = [v for v in x if v]
        return round(sum(x) / len(x), 1) if x else None
    res["stream_tok_s_before"], res["stream_tok_s_during"], res["stream_tok_s_after"] = avg(before), avg(during), avg(after)
    if res["stream_tok_s_before"] and res["stream_tok_s_during"]:
        res["slowdown_x"] = round(res["stream_tok_s_before"] / res["stream_tok_s_during"], 2)
    res["other_traffic_max_running"] = max((run for t, run, w in others if t_send <= t <= t_first), default=0) - 5
    res["stream_total_s"] = [round(r["done"] - r["sent"], 1) for r in S if "done" in r]
    print(json.dumps(res), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
