# Physical Intelligence｜荷崩れパレットの自律積み替え

輸送中に荷崩れしたパレットの荷を、門型の吸着ロボットが、別のパレットの決められた区画へ積み替えます。
NVIDIA Isaac Sim の物理シミュレーションの中で、観測・判断・行動をすべて自律で行います。

人が与えるのは「パレット A の荷を B へ積み替えて」という目的だけです。
どこから見るか、どの箱から取るか、置いた後に何を確かめるかは、ロボットが自分の見立てから決めます。

## この提出で確かめたこと

### 実演の走行（6 箱、全工程を省略なし）

| 項目 | 内容 |
|---|---|
| 荷 | 3 箱ずつ 2 段の 6 箱（`run_mission.py --grid 3 1 2`）。運搬中のブレーキ（4.49 m/s²）で上の段の 3 箱がずれた状態から始める（最大 0.18 m、床や段から落ちた箱なし） |
| 始まりの状態の選び方 | ロボットを動かす前に、荷崩れだけを物理で計算し、決めておいた 4 条件で番号 200 から順に判定した。最初に満たした 228 で 1 回走らせた（[docs/demo_load_selection.md](docs/demo_load_selection.md)、判定の記録は `artifacts/demo_survey/`） |
| 判断 | 真上のカメラの見立てから学んだ崩れの予測（`models/cascade_v4_camera.json`） |
| 結果 | 6 箱すべてを B へ積み替え、置いた後に自分のカメラで確かめた。二次の荷崩れ 0・運搬中の落下 0・人への引き継ぎ 0・置いた位置の誤差は最大 4.0 mm。作業の時間はシミュレーションの中で 300.7 秒 |
| 判定 | 段階 S3（`scripts/stage_gate.py`、基準は走らせる前から変えていない）に合格（`artifacts/demo_v3/gate.json`） |
| 映像 | 提出の映像は、下の通常の荷（24 箱）の走行。この 6 箱の走行も、同じ形の映像を作れる（`plans/submission_demo_2x.json`）。どちらも物理の刻みごとの記録から描き直し、2 倍速・省略なしで映す。横に監視画面を置き、各軸の位置、吸着と爪の力、手首の力の時系列、装置の状態、ロボットの目を、記録の値のまま時刻をそろえて出す（`plans/submission_demo_2x.json`、`scripts/console_view.py`） |
| 見た目の作り込み | 信号灯・制御盤・各軸のモーター・配線ダクトは描き直しの場面だけに置いた。走行の物理には入れていない（`shinai_pi/sim/visual_detail.py` が、物理の属性が変わっていないことを毎回確かめる） |

崩れの予測を使って取る順を決め、6 箱の積み替えに成功しました。
同じ番号は規則だけの判断でも成功しています（`artifacts/demo_v3/shinai_seed228`、誤差は最大 2.7 mm）。予測によって成功した、とは言えません。
規則だけの走行は、手順の取り違えで意図せず走ったものです。記録は消さずに残しています（違いは [docs/submission_scope.md](docs/submission_scope.md)）。

今回の成功は、6 箱・1 条件の 1 回です。
崩れ方を変えた別の 8 走行（24〜36 箱）では、全工程の基準（段階 S3）を満たしていません。詳細は [docs/other_runs.md](docs/other_runs.md) にあります。

### 提出の映像：通常の荷（24 箱）の最初の 3 分 57 秒

映像は、通常の荷（3 × 4 × 2 段＝24 箱、始まりの状態は番号 118）の走行の最初の 3 分 57 秒を、2 倍速・省略なしで映したものです（[docs/full_load_run.md](docs/full_load_run.md)、`plans/submission_full_2x.json`）。
この間にロボットは 5 箱を B へ移し、5 箱とも置いた後の確認に合格しています。
取る順・見に行く場所・運ぶ経路は、ロボットが走行中に自分で決めています。

この走行の全体は合格していません。6 箱目が置いた直後に倒れ、11 箱を置いたところで人の対応を待って止まりました（`artifacts/full_v3/`）。
置いた後の確認で「確かめられない」と出た箱を、置き直さずに次へ進んだことが原因の一つで、直す対象です。

### 機体：持ち上げた直後に爪を入れる

爪の台座を、吸着板の面にボルトで留める形に設計しました。
運ぶ順番は「持ち上げる → その場で爪を差し込む → その場で向きを合わせる → 爪で下から押さえる → 運ぶ」です。
揺れで落とさないよう、横へ動く前に爪で支えます。
動く範囲のどこでも部品どうしが重ならないことは、試験（`tests/test_structure.py`）で確かめています。

### AI：崩れの予測を、物理の試行から学ぶ

箱を一つ取り除いたとき、ほかの箱が崩れるかを予測します（ロジスティック回帰）。
学習のデータは、段の数・品目・ブレーキ・ラップの外れ方を変えた荷崩れの場面で、物理シミュレーションを試して作りました。
3 段の荷と 8.0 m/s² を超えるブレーキは学習に使わず、その条件で通用するかを確かめています。

| 予測 | 入力 | 学習に使っていない条件での AUC | 規則の AUC |
|---|---|---|---|
| `cascade_v4_camera`（実演の走行で使用） | 真上のカメラの深度と切り分けから作った見立て | 0.942 | 0.746 |
| `cascade_v3` | 正しい位置に雑音を足したもの | 0.937 | 0.566 |

AUC は、崩れる組み合わせを崩れない組み合わせより高く見積もれた割合です。1 に近いほど良く、0.5 は当て推量と同じです。
どちらも、データを作る前に決めた条件（AUC が規則より 0.10 以上高く、Brier スコアが規則以下）で合格しました（`artifacts/cascade_v3/`・`artifacts/cascade_v4_camera/`）。

**範囲と版は [docs/submission_scope.md](docs/submission_scope.md) にまとめています。**

## 何が独自か

以下は設計と実装の内容です。Isaac Sim での全工程の受け入れは、上の実演の走行の判定です。

**自分がまだ何を知らないかを数えて持ち、その不確かさが次の判断を変えうるときだけ、見に行く。**

- 見えていない空間を「空いている」と仮定しません。カメラの光線が通った場所・止まった場所・届いていない場所を分けて持ちます。
- 崩れた山の「誰が誰を支えているか」を推定します。ケースを 1 つ抜いたとき、どれが倒れるかを、落ちる高さの重みつきで予測します。
- 観測した山は静止している、という物理の前提を使います。推定の上で支えが足りないケースは「見えない支えがある」とみなし、むやみに危険と判断しません。
- 安全に取れるケースがあれば、上から取ります。なければ、判断を確定させるのに最も役立つ視点を、移動時間との釣り合いで選んで見に行きます。
- 取った後は、山を見直して予測と比べます。置く前には、置き場が空いていることをカメラで確かめます。
- **重心と重力。** 見えない中身の偏りを「箱の中心 ±60 mm」の幅で持ち、支えの判定に入れます。持ち上げた後は、手首の力覚（既知の荷重で校正）で質量と重心を測ります。
- **姿勢を戻してから積む。** 横倒し・立ったケースは手首で 90 度起こして小さな台に立て、天面で持ち直します。30 度まで傾いた面は、手首を面に合わせて吸着し、空中で水平に戻します。
- **正確なブロック配置。** 置いた後は、ケース自身の上面で位置・向き・傾きを測り（区画から 10 mm・2 度以内）、3 秒後にもう一度確かめます。ずれていれば掴み直して置き直します。
- **測れなかったことを測れなかったと言う。** 床のカメラがふさがれて下面が一部しか見えないとき、推定が隣と重なるときは、その値を使いません。積荷の個数と照合し、見つからないケースは人に報告します。
- 連鎖落下の予測が外れる場面を、物理シミュレーションの結果から学習します（摩擦と、ケースごとの重さ・重心を乱数で振ったデータ）。
- 安全監督は AI と独立に動きます。光線式安全装置（人の進入）、搬送中の落下、速度の超過で止まり、作業者の解除を待ちます。

設計の詳細は [docs/architecture.md](docs/architecture.md) にあります（書いた時点の対象は 1 パレット 48 箱です）。

## 動かす環境

| 項目 | この提出で使ったもの |
|---|---|
| OS | Windows 11 Pro |
| GPU | NVIDIA GeForce RTX 5070 Ti（16 GB）。Isaac Sim の走行で約 4 GB を使います |
| シミュレーター | NVIDIA Isaac Sim 6.2.0.0（`requirements-sim.txt`） |
| Python | 3.12.10 |
| 時間 | 実演の走行（6 箱）で約 9 分。24〜36 箱の走行は 1〜5 時間（2 視点の記録と物理の刻みの記録を含む。同じ PC で別の GPU の作業が動いていたため幅がある） |

```bat
py -3.12 -m venv .venv-isaac
.venv-isaac\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-isaac\Scripts\python -m pip install -r requirements-sim.txt --extra-index-url https://pypi.nvidia.com

py -3.12 -m venv .venv-dev
.venv-dev\Scripts\python -m pip install -r requirements-dev.lock
```

Isaac Sim は初回の起動時に NVIDIA Omniverse の利用許諾への同意を求めます（`OMNI_KIT_ACCEPT_EULA=YES`）。

## 動かし方

| やること | コマンド |
|---|---|
| 試験（シミュレーター不要） | `.venv-dev\Scripts\python -m pytest -q` |
| 実演の走行の始まりの状態を選ぶ | `.venv-isaac\Scripts\python scripts\survey_start_shift.py --grid 3 1 2 artifacts\demo_survey 200 201 … 239` |
| 実演の走行（6 箱） | `.venv-isaac\Scripts\python scripts\run_mission.py --grid 3 1 2 --seeds 228 --policies shinai_learned --cascade-model models\cascade_v4_camera.json --record --views presentation closeup --trace --out artifacts\demo_v3` |
| 実演の走行の判定 | `.venv-dev\Scripts\python scripts\stage_gate.py S3 artifacts\demo_v3` |
| 崩れ方を変えた場面の走行 | `.venv-isaac\Scripts\python scripts\run_mission.py --scenarios --seeds 3001 3014 3000 3025 --policies shinai_learned shinai --cascade-model models\cascade_v3.json --record --views presentation closeup --trace --out artifacts\missions_v3` |
| 段階 S3 の判定と表 | `.venv-dev\Scripts\python scripts\stage_gate.py S3 artifacts\missions_v3` と `.venv-dev\Scripts\python scripts\mission_table.py artifacts\missions_v3 --seeds 3001 3014 3000 3025` |
| 学習データ・学習・判定 | `scripts\gen_cascade_data.py --split train --seeds 2000 2399 --out artifacts\cascade_v3\train.npz`（seen・unseen も同様）、`scripts\train_cascade.py --data artifacts\cascade_v3 --model cascade_v3`、`scripts\judge_cascade.py artifacts\cascade_v3\eval.json` |
| ロボットの目の画像 | `.venv-dev\Scripts\python scripts\perception_views.py artifacts\missions_v3\shinai_learned_seed3001` |
| 開始状態の測定 | `.venv-isaac\Scripts\python scripts\survey_start_shift.py`（引数は [docs/start_state_selection.md](docs/start_state_selection.md)） |
| 映像を描き直す | `.venv-isaac\Scripts\python scripts\rerender_trace.py --run artifacts\demo_v3\shinai_learned_seed228 --grid 3 1 2 --uhd --fps 30 --speed 2 --from 4.0 --to 305.0 --out artifacts\video_frames\demo228_dressed3` |
| 映像を組み立てる | `.venv-dev\Scripts\python scripts\compose_video.py plans\submission_demo_2x.json` |
| 画面（Inspector） | `.venv-dev\Scripts\python -m shinai_pi.inspector_server` → http://127.0.0.1:8770/ |

描き直しには、走行の物理の刻みの記録（`--trace`）が要ります。容量のため、このリポジトリには入れていません。上の提出の走行で作り直せます。

## フォルダー

| 場所 | 中身 |
|---|---|
| `shinai_pi/` | 本体。司令塔・見立て・認識・観測計画・安全監督・窓口・評価・Inspector のサーバー |
| `shinai_pi/sim/` | Isaac Sim のセルの組み立てと実行 |
| `configs/cell.toml` | セルの寸法・ロボットの制限・ケースの寸法と重さ（仮の値）・輸送の条件 |
| `ui/` | Inspector の画面 |
| `tests/` | 試験 |
| `scripts/` | 実行・評価・動画 |
| `inspector/` | 前段の試作（MuJoCo の移動ロボットによる点検）。比較のために残している |

## 限界

- 物体の切り分けはシミュレーターの分割を使っています。姿勢・支え・見えないケースは距離画像から推定しています。
- 吸着・光線式安全装置はシミュレーションの近似です。吸着パッドの力は理想値（60 kPa・直径 80 mm）で、実物の定格は確かめていません。実機での性能は主張しません。
- 接触の瞬間の押し付け力には、剛な位置制御の模型に由来する山（1 ステップで約 1.8 kN）があります。静的な押し付けは 50 N 程度です。
- ケースの寸法・重さ・中身の偏りの範囲は仮の値です。
- 映像の文字の書体は Windows の場所（`C:/Windows/Fonts/`）を直接指しています。ほかの OS では書き換えが要ります。
- 各走行の `manifest.json` の `git_commit` は、開発用のリポジトリのコミットです。このリポジトリは履歴を持たない写しのため、同じハッシュはありません。

## Summary in English

Autonomous re-palletising of a transport-collapsed pallet in NVIDIA Isaac Sim. The robot keeps track of what it does not know
and goes to look only when that uncertainty could change its next decision.

Demonstration run: a six-case load (three cases on three) whose top layer slid by up to 0.18 m under braking. The start state
was chosen by a rule fixed before any robot run (seeds 200-239, first one meeting it: 228). The gantry moved all six cases to
pallet B and checked each with its own cameras: no secondary collapse, no drop, no hand-over, largest placement error 4.0 mm,
300.7 s of simulated time; it passed the stage S3 gate.
The submitted video shows the first 3 min 57 s of a normal 24-case run (start state 118) at 2x with nothing cut, beside a
monitoring screen read from the run's trace: five cases were placed and verified in that time. That run as a whole did not
pass: the sixth case toppled after release and the robot stopped after 11 cases, waiting for a person.

AI: a collapse predictor learned from physics trials (logistic regression). The version fed by the overhead camera's belief
was used in the run; on collapse conditions kept out of training it reached AUC 0.94 against 0.75 for the geometric rule.
The fork mount is bolted onto the suction plate and the tines go under a case right after the lift, before any sideways move.

Simulation only; object segmentation comes from the simulator. Other runs: docs/other_runs.md. Scope: docs/submission_scope.md.

## ライセンス

MIT License（[LICENSE](LICENSE)）。NVIDIA Isaac Sim は NVIDIA のライセンスに従います。本リポジトリは Isaac Sim の資産を含みません（セルの形はすべてコードで組み立てています）。
