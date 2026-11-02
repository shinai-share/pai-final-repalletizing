"""Local-only console. No arbitrary commands, file serving, or external services."""
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import json
from .run import run

class Handler(BaseHTTPRequestHandler):
    def send(self,status,body,kind='application/json; charset=utf-8'):
        payload=body.encode('utf-8');self.send_response(status)
        self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)))
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/':self.send(200,Path(__file__).with_name('console.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
        elif self.path=='/scene.png':
            payload=Path('artifacts/3d-scene.png').read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        else:self.send(404,'{}')
    def do_POST(self):
        if self.path!='/api/run':return self.send(404,'{}')
        origin=self.headers.get('Origin','')
        if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):return self.send(403,'{}')
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<4096:raise ValueError('入力が長すぎます')
            a=json.loads(self.rfile.read(size))
            if a.get('scenario') not in ['normal','blocked','sensor_dropout','combined','no_route']:raise ValueError('未対応の条件')
            seed=int(a.get('seed',42))
            if not 0<=seed<=100000:raise ValueError('シードは0〜100000です')
            result=run(a['scenario'],seed,a.get('instruction',''),capture_preview=True)
            self.send(200,json.dumps(result,ensure_ascii=False))
        except (ValueError,TypeError) as e:self.send(400,json.dumps({'error':str(e)},ensure_ascii=False))
    def log_message(self,*args):pass

if __name__=='__main__':
    print('Local console: http://127.0.0.1:8765',flush=True)
    HTTPServer(('127.0.0.1',8765),Handler).serve_forever()
