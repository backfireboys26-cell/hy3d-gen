"""ping.py <expected_code> [url] - one GET of the gate's health path; prints 'PING <code> <body>'
and exits 0 only when the code matches. Used inside the test container (no curl there)."""
import sys
import urllib.error
import urllib.request

want = int(sys.argv[1])
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080/ping"
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        code, body = r.status, r.read()[:300].decode(errors="replace")
except urllib.error.HTTPError as e:
    code, body = e.code, e.read()[:300].decode(errors="replace")
except Exception as e:
    code, body = 0, repr(e)
print(f"PING {code} {body}", flush=True)
sys.exit(0 if code == want else 1)
