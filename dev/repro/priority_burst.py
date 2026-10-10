"""Sub-agent priority: 6 sub-agent-marked requests (fresh 20k prompts) and 1 interactive request (fresh 8k) arrive
together; report the interactive request's TTFT and the sub-agents' median TTFT. Needs --scheduling-policy priority and the
marker knobs (see experimental/compose/adaptive-k.yaml); GLM_URL defaults to http://127.0.0.1:8888."""
import json, os, random, sys, threading, time, urllib.request
URL = os.environ.get("GLM_URL", "http://127.0.0.1:8888") + "/v1/chat/completions"
label = sys.argv[1]; rnd = random.Random()
W = "river stone lantern harbor meadow copper falcon willow ember signal orchard canyon glacier thunder".split()
def text(n): return " ".join(rnd.choice(W) for _ in range(int(n / 1.25)))
MARK = "You are a focused subagent working on a specific delegated task."
res = {}
def go(name, sysmsg, n):
    body = {"model": "GLM-5.3-Flash-EXL3", "stream": True, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "system", "content": sysmsg + "\n" + text(n)}, {"role": "user", "content": "Reply READY."}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            l = raw.decode().strip()
            if l.startswith("data: ") and l != "data: [DONE]" and first is None:
                d = json.loads(l[6:])
                if any((c.get("delta") or {}).get(k) for c in d.get("choices") or [] for k in ("content", "reasoning_content", "reasoning")):
                    first = time.time()
    res[name] = round((first or time.time()) - t0, 2)
th = [threading.Thread(target=go, args=("sub%d" % i, MARK + " Task %d." % i, 20000)) for i in range(6)]
for t in th: t.start()
time.sleep(0.05)
ti = threading.Thread(target=go, args=("interactive", "You are a helpful assistant.", 8000)); ti.start()
for t in th + [ti]: t.join()
subs = sorted(v for k, v in res.items() if k.startswith("sub"))
r = {"label": label, "interactive_ttft": res["interactive"], "sub_ttft_median": subs[3], "sub_ttft_max": subs[-1]}
print(json.dumps(r)); open("priotest.jsonl", "a").write(json.dumps(r) + "\n")
