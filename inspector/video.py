"""Recorded MuJoCo frames with truthful telemetry, Japanese explanatory cards."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

BG='#0b1420'; FG='#edf5fc'; MUTED='#92a9bf'; CYAN='#43e0cf'

def font(size):
    for p in ['C:/Windows/Fonts/YuGothM.ttc','C:/Windows/Fonts/meiryo.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default(size=size)

class Recorder:
    fps=20
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.writer=imageio.get_writer(path,fps=self.fps,codec='libx264',quality=8,macro_block_size=2,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        self.frames=0
        self.card('INSPECTION INTELLIGENCE', '現場の変化に、根拠を持って対応する。',[
            '自律設備点検のための独自制御基盤',
            '日本語指示 → 3地点の点検 → 異常表示の検出 → 報告',
            '途中で通路を封鎖。センサー通信も一時停止。',
            'ロボットは何を根拠に止まり、どう復旧するのか。'],14)
        self.card('ARCHITECTURE / 独自実装', '判断と安全を、別の仕組みで支える。',[
            '認知：距離センサー + 3D搭載カメラ画像から学習した認識',
            '世界状態：観測を蓄積する占有地図 / 経路計画：A*',
            '安全監督：鮮度・制動距離を確認し、速度指令を遮断',
            '復旧：再観測 → 再計画。通行不能なら停止して中断。'],14)

    def append(self,im):
        self.writer.append_data(np.asarray(im));self.frames+=1

    def card(self,label,title,lines,seconds):
        im=Image.new('RGB',(1440,900),BG);d=ImageDraw.Draw(im)
        d.rectangle((65,75,70,825),fill=CYAN)
        d.text((100,90),label,font=font(26),fill=CYAN)
        d.text((100,210),title,font=font(43),fill=FG)
        for i,line in enumerate(lines):d.text((100,355+i*83),line,font=font(28),fill=MUTED)
        d.text((100,815),'Physical AI research prototype',font=font(22),fill=MUTED)
        for _ in range(int(seconds*self.fps)):self.append(im)

    def add(self,plant,brain,row):
        im=Image.new('RGB',(1440,900),BG);im.paste(Image.fromarray(plant.render()),(0,110))
        d=ImageDraw.Draw(im)
        d.text((35,25),'自律点検シミュレーション',font=font(31),fill=FG)
        d.text((1030,30),'物理シミュレーション 2倍速',font=font(23),fill=CYAN)
        d.line((980,120,980,810),fill='#294253',width=2)
        x=1010
        d.text((x,130),f"SIM TIME  {row['t']:05.1f} s",font=font(30),fill=FG)
        d.text((x,195),f"目標  {brain.target}",font=font(27),fill=CYAN)
        state={'NAVIGATE':'走行・経路追従','INSPECT':'停止・3回の再観測','RECOVERY':'安全停止・復旧待機','COMPLETE':'点検完了','ABORTED':'安全中断'}[brain.state]
        d.text((x,250),state,font=font(26),fill=FG)
        d.text((x,305),f"安全監督  {brain.reason}",font=font(22),fill=MUTED)
        d.text((x,350),f"再計画 {brain.replans} 回 / 衝突 {plant.collisions} 回",font=font(24),fill=FG)
        d.text((x,410),'点検結果',font=font(23),fill=MUTED)
        for i,f in enumerate(brain.findings):
            label={'ANOMALY':'異常表示','NORMAL':'正常表示','REVIEW':'要確認'}[f['result']]
            d.text((x,455+i*42),f"{f['target']}  {label}",font=font(25),fill='#ffad76' if f['result']=='ANOMALY' else CYAN)
        if brain.last_image is not None:
            im.paste(Image.fromarray(brain.last_image).resize((130,130)),(x,625))
            d.text((x+150,640),'搭載カメラRGB',font=font(20),fill=MUTED)
            d.text((x+150,678),'3D画像で学習済み',font=font(20),fill=MUTED)
        note='距離観測から地図を更新し、点検地点へ向かいます。'
        if brain.reason=='SENSOR_HOLD':note='通信途絶を検知。古い観測では走行せず、新しい観測を待ちます。'
        elif brain.state=='INSPECT':note='到達後に3回観測。判断が揃わなければ「要確認」として報告します。'
        elif plant.gate_closed and row['t']<25:note='オレンジの通路封鎖を距離センサーで検知。上側の経路へ迂回します。'
        d.rectangle((0,830,1440,900),fill='#142638')
        d.text((30,850),note,font=font(26),fill=FG)
        self.append(im)

    def finish(self,r):
        self.card('MEASURED RESULT / この実行の結果','点検・迂回・停止・復旧を一つの系で検証。',[
            f"3地点完了：{r['completed']}    衝突：{r['collisions']}回    模擬時間：{r['sim_seconds']:.1f}秒",
            f"再計画：{r['replans']}回    制御判断 p95：{r['control_latency_ms_p95']:.2f} ms",
            '観測画像・時刻・判断理由をJSONとPNGへ保存。',
            '3D空間の照明・視点を変えて撮影し、再学習できる構成。'],14)
        self.card('SCOPE / 検証範囲を明確に','研究用の簡略化モデルです。',[
            '3D剛体・左右輪駆動・摩擦・重力を計算。実機は未検証。',
            '自己位置は模擬推定値。RGBは搭載カメラの実描画。',
            '実設備の故障診断や、実世界での安全性を保証しません。',
            '次段階：実カメラ・自己位置推定・実機モデル同定。'],13)

    def close(self):
        if self.writer:self.writer.close();self.writer=None
