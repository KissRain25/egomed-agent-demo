# -*- coding: utf-8 -*-
"""
EgoMed local proxy for Gradio: 解决 gradio 6.26 前端 JS 从 AWS S3 (gradio.s3-us-west-2.amazonaws.com)
拉取被墙导致网页黑屏的问题。

原理: gradio 服务 (默认 127.0.0.1:7860) 的本地 /assets/ 其实已托管完整前端 bundle,
只是首页 HTML 把 <script src> 指向 S3。本代理监听 127.0.0.1:7861, 转发所有请求到 7860,
并仅对 HTML 响应做字符串重写: S3 地址 -> 本地 /assets/ 地址。

用法:
    python gradio_local_proxy.py [--listen 127.0.0.1] [--port 7861] [--upstream 127.0.0.1:7860]
然后浏览器打开 http://127.0.0.1:7861
"""
import argparse
import http.client
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gradio

GRADIO_FRONTEND_ASSETS = (
    Path(gradio.__file__).resolve().parent / "templates" / "frontend" / "assets"
)

CDN_ROOT = "https://gradio.s3-us-west-2.amazonaws.com/6.26.0/"

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def rewrite_html(body: bytes) -> bytes:
    """把 HTML 中指向 S3 CDN 的资源地址替换为本地 /assets/ 地址。"""
    html = body.decode("utf-8", errors="replace")
    # 主入口 gradio.js -> 本地 assets 的 module entry (与本地模板 ./assets/index-DZobeNVa.js 对应)
    html = html.replace(
        CDN_ROOT + "gradio.js",
        "/assets/index-DZobeNVa.js",
    )

    def _repl(match: re.Match) -> str:
        name = match.group(1)
        if (GRADIO_FRONTEND_ASSETS / name).exists():
            return "/assets/" + name
        return match.group(0)  # 本地没有同名资源则保留原样

    html = re.sub(
        re.escape(CDN_ROOT) + r"([A-Za-z0-9._-]+\.(?:js|css|json|png|svg|woff2?))",
        _repl,
        html,
    )
    return html.encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_host = "127.0.0.1"
    upstream_port = 7860

    # ---- helpers ---------------------------------------------------
    def _forward(self, method: str) -> None:
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=None)
        try:
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in HOP_BY_HOP and k.lower() != "host"
            }
            headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
            body = None
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                body = self._read_request_body()
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001
            self.send_error(502, f"upstream error: {exc}")
            conn.close()
            return

        try:
            self.send_response_only(resp.status)
            ctype = resp.getheader("Content-Type", "")
            is_html = ctype.startswith("text/html")
            is_sse = ctype.startswith("text/event-stream")

            # 转发响应头(去掉 hop-by-hop 与 Transfer-Encoding/Content-Length, 稍后自己决定)
            for key, value in resp.getheaders():
                kl = key.lower()
                if kl in HOP_BY_HOP or kl in ("content-length",):
                    continue
                self.send_header(key, value)

            if is_html:
                data = resp.read()
                data = rewrite_html(data)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif is_sse:
                # SSE: 无 Content-Length, 流式转发直到上游结束
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 浏览器已离开
            else:
                # 带 Content-Length 的普通资源(图片等), 一次性读完转发
                if resp.length and resp.length > 0:
                    data = resp.read(resp.length)
                else:
                    data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            self.close_connection = True
        finally:
            conn.close()

    def _read_request_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length and length.isdigit():
            return self.rfile.read(int(length))
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = b""
            while True:
                line = self.rfile.readline().strip()
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline()
                    break
                data += self.rfile.read(size)
                self.rfile.read(2)
            return data
        return None

    # ---- verbs -----------------------------------------------------
    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_PUT(self):
        self._forward("PUT")

    def do_PATCH(self):
        self._forward("PATCH")

    def do_DELETE(self):
        self._forward("DELETE")

    def do_HEAD(self):
        self._forward("HEAD")

    def do_OPTIONS(self):
        self._forward("OPTIONS")

    def log_message(self, fmt, *args):  # 静默访问日志, 减少刷屏
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--upstream", default="127.0.0.1:7860")
    args = ap.parse_args()

    host, _, port_str = args.upstream.partition(":")
    ProxyHandler.upstream_host = host
    ProxyHandler.upstream_port = int(port_str or 7860)

    server = ThreadingHTTPServer((args.listen, args.port), ProxyHandler)
    print(
        f"[gradio-local-proxy] listening on http://{args.listen}:{args.port} "
        f"-> http://{host}:{ProxyHandler.upstream_port} (assets from local /assets/)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
