"""Which certificates does this machine see on HTTPS? For each host: a verified handshake with the default context (what
urllib uses), and, when it fails, the subject and issuer of each certificate presented, read from a second handshake
that does not verify. Only handshakes: no request is sent and nothing is written. Proxy variables: names and proxy hosts
only.

    python outputs/host_ops/tls_probe.py [HOST ...]
"""
import os
import socket
import ssl
import sys
import tempfile
import urllib.parse
import urllib.request

HOSTS = sys.argv[1:] or ["huggingface.co", "hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com",
                         "cas-bridge.xethub.hf.co", "pypi.org", "github.com", "www.google.com"]


def name(d):
    return ", ".join(f"{k}={v}" for rdn in d for k, v in rdn)


def decode(der):
    try:
        fd, p = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(fd, "w") as f:
            f.write(ssl.DER_cert_to_PEM_cert(der))
        try:
            return ssl._ssl._test_decode_cert(p)
        finally:
            os.unlink(p)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


print("python", sys.version.split()[0], "|", ssl.OPENSSL_VERSION, "|", sys.executable)
print("proxies:", {k: urllib.parse.urlsplit(v).hostname for k, v in urllib.request.getproxies().items()} or "none")
print("default verify paths:", ssl.get_default_verify_paths())
for h in HOSTS:
    try:
        with socket.create_connection((h, 443), timeout=15) as s:
            with ssl.create_default_context().wrap_socket(s, server_hostname=h) as t:
                print(f"{h}: verified | issuer {name(t.getpeercert()['issuer'])}")
                continue
    except ssl.SSLCertVerificationError as e:
        err = f"{e.verify_message} (code {e.verify_code})"
    except Exception as e:  # noqa: BLE001
        print(f"{h}: {type(e).__name__}: {e}")
        continue
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((h, 443), timeout=15) as s:
            with ctx.wrap_socket(s, server_hostname=h) as t:
                chain = t.get_unverified_chain() if hasattr(t, "get_unverified_chain") else [t.getpeercert(True)]
        print(f"{h}: NOT verified: {err} | {len(chain)} certificate(s) presented")
        for i, der in enumerate(chain):
            info = decode(der)
            if "error" in info:
                print(f"   [{i}] {info['error']}")
            else:
                print(f"   [{i}] subject {name(info['subject'])} | issuer {name(info['issuer'])} "
                      f"| {info.get('notBefore')} to {info.get('notAfter')}")
    except Exception as e:  # noqa: BLE001
        print(f"{h}: NOT verified: {err} | chain unreadable: {type(e).__name__}: {e}")
