# cascade_v4_camera の学習データ（用途を限ったもの）

- 作り方: `scripts/gen_cascade_data.py --input camera --split train|seen|unseen`（場面は番号 2000〜2399、unseen は 2000〜2120 の 60 個）
- 入力: 真上のカメラ（overhead_a）の深度と物体ごとの切り分けから、走行の始めと同じ処理（`Executive.look`）で作った見立ての特徴
- 教師: 箱を一つ作業場の外へ瞬時に移し、1.5 秒後にほかの箱が動いたか（cascade_v3 と同じ）
- 計画: `docs/camera_learning_plan.md`（データを作る前に固定）

限り: 物体ごとの切り分けはシミュレーターが出したもの。見るのは真上の固定カメラ一つ。カメラに写らない箱は特徴も教師も作られない。
この予測は、提出の走行では使っていない。走行で効くかは未確認。
