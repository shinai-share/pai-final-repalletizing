"""Metric occupancy planning; no simulator state is accessed here."""
import heapq
import math
import re
import numpy as np

TARGETS = {'ポンプA': (4., -2.), '配電盤B': (4., 2.), 'ファンC': (-4., 2.)}
WALLS = [(0., 0., .16, 1.15), (0., -3.65, .16, .35), (0., 3.65, .16, .35)]

def parse_instruction(text):
    """Conservative controlled Japanese grammar. Ambiguous inputs never actuate."""
    if not isinstance(text, str) or len(text) > 300:
        raise ValueError('指示は300文字以内で入力してください')
    if any(w in text for w in ('無視', '解除', '衝突', '最速', '停止しない', '点検しない')):
        raise ValueError('安全条件を変更する指示、否定指示は受理できません')
    if '点検' not in text:
        raise ValueError('点検する対象を指定してください')
    names = sorted((n for n in TARGETS if n in text), key=text.index)
    if not names:
        raise ValueError('対象: ポンプA、配電盤B、ファンC')
    residual = text
    for n in names:
        residual = residual.replace(n, '')
    for phrase in ('を順に点検して', 'を点検してください', 'を点検して', 'を点検', '点検', 'してください', 'して', '順に', '、', 'と', ' ', '。', '→', ',', '，'):
        residual = residual.replace(phrase, '')
    if residual.strip():
        raise ValueError('未対応の表現です。「ポンプA、配電盤B、ファンCを順に点検して」を使用してください')
    return names

class WorldModel:
    resolution = .25
    radius = .48  # conservative footprint including control error

    def __init__(self):
        self.dynamic = set()
        self.revision = 0
        self.static = set()
        for i in range(49):
            for j in range(33):
                x, y = self.xy((i,j))
                if abs(x) > 5.4 or abs(y) > 3.4 or any(abs(x-a) < sx+self.radius and abs(y-b) < sy+self.radius for a,b,sx,sy in WALLS):
                    self.static.add((i,j))

    def cell(self, p):
        return (int(round((p[0]+6)/self.resolution)), int(round((p[1]+4)/self.resolution)))

    def xy(self, c):
        return (c[0]*self.resolution-6, c[1]*self.resolution-4)

    def observe(self, points):
        before = len(self.dynamic)
        for x,y in points:
            c = self.cell((x,y))
            for di in range(-2,3):
                for dj in range(-2,3):
                    cell = (c[0]+di,c[1]+dj)
                    if cell not in self.static and 0 <= cell[0]<49 and 0<=cell[1]<33:
                        if math.dist(self.xy(cell),(x,y)) <= self.radius+.15:
                            self.dynamic.add(cell)
        if len(self.dynamic) != before:
            self.revision += 1
        return len(self.dynamic) != before

    def plan(self, start, goal):
        s,g = self.cell(start), self.cell(goal)
        blocked = self.static | self.dynamic
        if g in blocked:
            return []
        # Allow the current cell as a seed; do not allow expansion through obstacles.
        frontier = [(0.,s)]
        cost, parent = {s:0.}, {}
        while frontier:
            _, c = heapq.heappop(frontier)
            if c == g:
                route = [c]
                while c != s:
                    c = parent[c]; route.append(c)
                return [self.xy(q) for q in reversed(route)]
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                n = (c[0]+di,c[1]+dj)
                if n in blocked or not (0 <= n[0]<49 and 0 <= n[1]<33):
                    continue
                v = cost[c]+1
                if v < cost.get(n,float('inf')):
                    cost[n]=v; parent[n]=c
                    heapq.heappush(frontier,(v+abs(n[0]-g[0])+abs(n[1]-g[1]),n))
        return []

def supervise(v, w, ranges, age, measured_speed, emergency=False):
    """Independent, fail-closed velocity envelope; range measured from chassis center."""
    r = np.asarray(ranges)
    if emergency:
        return 0.,0.,'EMERGENCY_STOP'
    if not math.isfinite(age) or not 0<=age<=.25 or len(r) != 72 or not np.isfinite(r).all() or not math.isfinite(measured_speed):
        return 0.,0.,'SENSOR_HOLD'
    if not math.isfinite(v) or not math.isfinite(w):
        return 0.,0.,'COMMAND_HOLD'
    clearance = float(r.min())-.36
    stop_distance = measured_speed**2/(2*1.5)+abs(measured_speed)*.10+.12
    if clearance < stop_distance:
        return 0.,0.,'OBSTACLE_HOLD'
    return float(np.clip(v,0,.65)), float(np.clip(w,-1.5,1.5)), 'CLEAR'
