import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from shinai_pi import inspector_server as srv


@pytest.fixture()
def server(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    run = runs / 'shinai_seed001'
    (run / 'frames').mkdir(parents=True)
    (run / 'frames' / '00001000.jpg').write_bytes(b'\xff\xd8jpeg')
    (run / 'events.jsonl').write_text('\n'.join(json.dumps({'seq': i, 't': i, 'type': 'OBSERVED', 'data': {}}) for i in range(3)) + '\n{"seq": 3, "t"', encoding='utf-8')
    (tmp_path / 'secret.txt').write_text('no', encoding='utf-8')
    monkeypatch.setattr(srv, 'RUNS', runs)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), srv.Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b''


def test_lists_runs_and_reads_partial_live_log(server):
    s, body = get(server + '/api/runs')
    assert s == 200 and json.loads(body)[0]['status'] == 'running'
    s, body = get(server + '/api/run/shinai_seed001/events?since=0')
    assert [e['seq'] for e in json.loads(body)] == [1, 2]          # the half-written line is skipped


def test_serves_frames_and_the_screen(server):
    assert get(server + '/api/run/shinai_seed001/frame/00001000.jpg') == (200, b'\xff\xd8jpeg')
    assert get(server + '/')[0] == 200


@pytest.mark.parametrize('path', ['/api/run/..%2Fsecret/events', '/api/run/shinai_seed001/frame/..%2F..%2Fsecret.txt',
                                  '/api/run/shinai_seed001/frame/1.jpg', '/secret.txt', '/api/run/SHINAI_seed001/events'])
def test_nothing_outside_the_run_files_is_served(server, path):
    assert get(server + path)[0] == 404


def test_serves_the_side_and_close_up_views_and_lists_them(server, tmp_path):
    run = srv.RUNS / 'shinai_seed001'
    (run / 'frames_side').mkdir()
    (run / 'frames_side' / '00001000.jpg').write_bytes(b'\xff\xd8side')
    assert json.loads(get(server + '/api/run/shinai_seed001/views')[1]) == ['side']
    assert get(server + '/api/run/shinai_seed001/frame/side/00001000.jpg') == (200, b'\xff\xd8side')
    assert get(server + '/api/run/shinai_seed001/frame/closeup/00001000.jpg')[0] == 404
    assert get(server + '/api/run/shinai_seed001/frame/..%2Fframes/00001000.jpg')[0] == 404
    assert get(server + '/api/run/shinai_seed001/frame/secret/00001000.jpg')[0] == 404


def test_serves_what_the_robot_saw_and_made_of_it(server):
    # owner 2026-10-31: the screen shows the robot's own camera with its estimate drawn on it (perception_views.py)
    pd = srv.RUNS / 'shinai_seed001' / 'perception'
    pd.mkdir()
    (pd / 'index.json').write_text(json.dumps({'views': [{'t': 5.0, 'picture': '00005000_overhead_a_view.jpg'}]}),
                                   encoding='utf-8')
    (pd / '00005000_overhead_a_view.jpg').write_bytes(b'\xff\xd8eye')
    assert json.loads(get(server + '/api/run/shinai_seed001/perception')[1])['views'][0]['t'] == 5.0
    assert get(server + '/api/run/shinai_seed001/perception/00005000_overhead_a_view.jpg') == (200, b'\xff\xd8eye')
    assert get(server + '/api/run/shinai_seed001/perception/..%2Fevents.jsonl')[0] == 404
    assert get(server + '/api/run/shinai_seed001/perception/index.json')[0] == 404


def test_a_run_without_camera_records_has_no_views(server):
    assert json.loads(get(server + '/api/run/shinai_seed001/perception')[1]) == {'views': []}


def test_runs_of_four_digit_scenario_seeds_are_listed(server):
    (srv.RUNS / 'shinai_learned_seed3001').mkdir()
    assert 'shinai_learned_seed3001' in [r['name'] for r in json.loads(get(server + '/api/runs')[1])]
