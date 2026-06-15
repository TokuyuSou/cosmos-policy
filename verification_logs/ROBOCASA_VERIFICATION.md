# Cosmos Policy × RoboCasa 動作確認レポート（再現手順つき）

**目的:** `cosmos-policy` リポジトリの README / SETUP.md / ROBOCASA.md に従い、RoboCasa シミュレーション
ベンチマークでの評価（eval）が **正しく走ること** を確認する。少量サブセット（2 トライアル）で実施。

**結果:** ✅ 成功。`TurnOffMicrowave` タスクで 2/2 episode が成功し、評価パイプラインが
エラーなく完走（exit code 0）。

| 項目 | 値 |
|---|---|
| 実施日 | 2026-06-15 |
| タスク | `TurnOffMicrowave`（指示文: "press the stop button on the microwave"） |
| トライアル数 | 2（README デフォルトの 50 を縮小） |
| Success rate | 1.0000 (100%, 2/2) |
| 平均 episode 長 | 279.5 steps |
| 生成物 | 結果ログ `.txt` 1 件 + ロールアウト動画 `.mp4` 4 件 |
| GPU | NVIDIA RTX A6000 (48GB) を 1 枚使用（`CUDA_VISIBLE_DEVICES=0`） |

---

## 0. 実行環境（前提）

この作業は **既に起動済みの Docker コンテナ内**（cosmos-policy イメージ）で完結している。
`docker build` / `docker run` 自体は事前に済んでおり、SETUP.md のコンテナは稼働中だった。

- コンテナ確認: `/.dockerenv` 存在、`whoami`=`root`、hostname=`103eff182fe0`
- ホストからのマウント（= 「今マウントされているディレクトリ」。ここ以外には一切書き込んでいない）:
  - `/workspace`  ← ホスト `/data1/cao`（`docker_run.sh` 定義）
  - `/home/cao/.cache` ← ホスト `/data1/cao/.cache`（各種キャッシュ）
  - 上記以外（`/root`, `/tmp` 等）はコンテナの overlay FS で ephemeral。ホストに影響しない。
- コンテナに事前設定済みの重要な環境変数（`docker_run.sh` 由来。**ヘッドレス MuJoCo 描画に必須**）:
  ```
  MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl
  HF_HOME=/home/cao/.cache/huggingface
  HUGGINGFACE_HUB_CACHE=/home/cao/.cache/huggingface/hub
  UV_CACHE_DIR=/home/cao/.cache/uv
  TORCH_HOME=/home/cao/.cache/torch
  NVIDIA_DRIVER_CAPABILITIES=all
  ```

### バージョン（`uv sync` 後に確定したもの）

| ツール / パッケージ | バージョン |
|---|---|
| Python | 3.10.18 |
| uv | 0.8.12 |
| torch | 2.7.0+cu128 (CUDA 12.8 build) |
| robosuite | 1.5.1 |
| mujoco | 3.2.6 |
| robocasa | 0.2.0（`moojink/robocasa-cosmos-policy`, commit `edd9a32`） |
| numpy | 2.2.6 |
| cosmos-policy リポジトリ | commit `1b6a685`（fork: `TokuyuSou/cosmos-policy`） |
| ドライバ / GPU | NVIDIA 535.230.02 / RTX A6000 × 3 |

---

## 1. 再現手順（コマンド）

すべて作業ディレクトリ `/workspace/cosmos-policy` で実行する。

### 1-1. RoboCasa 依存関係のインストール（ROBOCASA.md 通り）

```bash
cd /workspace/cosmos-policy
uv sync --extra cu128 --group robocasa --python 3.10
```
→ `.venv/`（約 13GB）が作成される。所要 ~数分。

### 1-2. RoboCasa パッケージのクローン & インストール（ROBOCASA.md 通り）

```bash
# README は `git clone ...`（フル）だが、本検証では速度のため --depth 1 を使用（任意）
git clone --depth 1 https://github.com/moojink/robocasa-cosmos-policy.git
uv pip install -e robocasa-cosmos-policy
```
> 補足: `uv pip install -e` は numba/llvmlite/protobuf 等を robocasa 要件に合わせてダウングレードする。
> その後の `uv run --group robocasa` 実行でも editable install は除去されず、`import robocasa` は成功する（確認済み）。

### 1-3. キッチンアセットのダウンロード（ROBOCASA.md 通り、~8GB）

```bash
# スクリプトは "~5 Gb のデータをDLします。Proceed? (y/n)" と1回だけ対話プロンプトを出すので y を送る
printf 'y\n' | uv run --extra cu128 --group robocasa --python 3.10 \
  robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
```
→ textures / fixtures / objaverse / generative_textures を
`robocasa-cosmos-policy/robocasa/models/assets/` 配下に展開（aigen_objs はスクリプト側で skip）。

### 1-4. private macros のセットアップ（ROBOCASA.md 通り）

```bash
uv run --extra cu128 --group robocasa --python 3.10 \
  robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
```
→ `robocasa/macros_private.py` を生成。

### 1-5. ⚠️ 追加で必要だった操作: Hugging Face トークン（README に記載なし）

評価で使う推論 config（`cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference`）は、
**gated（アクセス制限付き）リポジトリ** `nvidia/Cosmos-Predict2-2B-Video2World` を参照する:

- `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:148`
  `load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt")`
  （config の import 時に**即時ダウンロード**を試みる。base モデル 3.9GB）
- `cosmos_policy/config/defaults/tokenizer.py:27`
  `vae_pth="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"`
  （**推論で実際に使う** VAE トークナイザ 508MB）

トークン無しで実行すると以下で停止する:
```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo ... nvidia/Cosmos-Predict2-2B-Video2World ...
```

**対処:** gated repo へのアクセス権を持つ HF アカウントのトークンを用意し、
そのアカウントで https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World の利用規約に同意した上で、
環境変数 `HF_TOKEN` を設定する（`huggingface_hub` が自動で参照する）。

```bash
# 本検証ではユーザー提供トークンを使用（値は秘匿）。トークンはホストに永続化しないため、
# マウント外の ephemeral パス /root/.hftoken (mode 600) に置き、env 変数経由でのみ渡した。
export HF_TOKEN="<ACCESS_TO_nvidia/Cosmos-Predict2-2B-Video2World を持つトークン>"
```
> 注: `huggingface-cli login` を使うとトークンが `/home/cao/.cache/huggingface/token`（= マウント、ホストに永続）に
> 書かれる。本検証では永続化を避けるため `HF_TOKEN` env 変数のみで運用した。

> ※ 本来 base モデル(`load_path`)は推論時に fine-tuned checkpoint で上書きされる（`model_loader.py:90-92`）ため
> 重みとしては不要。しかし config import 時に即 DL を試みる実装のため、結局アクセス権が必要。
> tokenizer VAE の方は推論で実使用するため、いずれにせよ gated repo アクセスは必須。

### 1-6. 評価の実行（少量サブセット: 2 トライアル）

ROBOCASA.md の評価コマンドそのままで、`--num_trials_per_task` のみ 50→2 に縮小:

```bash
cd /workspace/cosmos-policy
export HF_TOKEN="<...>"            # 1-5 のトークン
CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --group robocasa --python 3.10 \
  python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --num_wrist_images 1 \
    --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave \
    --num_trials_per_task 2 \
    --run_id_note smoketest--2trials--seed195--deterministic \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 --randomize_seed False --deterministic True \
    --use_variance_scale False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
    --data_collection False
```

初回実行時、以下が自動 DL される（HF_HOME=`/home/cao/.cache/huggingface` 配下）:
- `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`（fine-tuned ckpt + dataset stats + T5 embeddings, 約 4.1GB）
- `nvidia/Cosmos-Predict2-2B-Video2World`（gated: base モデル 3.9GB + tokenizer VAE 508MB）

---

## 2. 実行結果

```
Starting evaluation for task: TurnOffMicrowave
Number of trials: 2
  Episode 0: SUCCESS (length: 280)   # Success detected at timestep 279
  Episode 1: SUCCESS (length: 279)   # Success detected at timestep 278
================================================================================
FINAL RESULTS
================================================================================
Task: TurnOffMicrowave
Success rate: 1.0000 (100%)
Average episode length: 279.5
Total episodes: 2
Total successes: 2
=== EVAL exit code: 0 ===
```

### 生成物（すべてマウント内）
- 結果ログ:
  `cosmos_policy/experiments/robot/robocasa/logs/ENV_EVAL-TurnOffMicrowave-cosmos-2026_06_15-07_21_05--smoketest--2trials--seed195--deterministic.txt`
- ロールアウト動画（episode ごとに 2 種: 通常 / future-image 予測つき）:
  `cosmos_policy/experiments/robot/robocasa/logs/rollout_data/TurnOffMicrowave--2026_06_15-07_21_05/*.mp4`

所要時間の目安（RTX A6000 1 枚）: モデルロード ~1.5 分 + 2 episode ~2.5 分 = 合計 **~4 分**。

---

## 3. ダウンロード/生成データの配置（マウント内のみ）

| 内容 | パス | サイズ | マウント |
|---|---|---|---|
| RoboCasa キッチンアセット | `/workspace/cosmos-policy/robocasa-cosmos-policy/robocasa/models/assets/` | 8.2 GB | `/workspace` |
| robocasa ソース(git clone) | `/workspace/cosmos-policy/robocasa-cosmos-policy/` | — | `/workspace` |
| Python venv | `/workspace/cosmos-policy/.venv` | 13 GB | `/workspace` |
| 評価ログ + 動画 | `/workspace/cosmos-policy/cosmos_policy/experiments/robot/robocasa/logs/` | <2 MB | `/workspace` |
| HF: fine-tuned ckpt | `/home/cao/.cache/huggingface/.../models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B/` | 4.1 GB | `/home/cao/.cache` |
| HF: gated base+tokenizer | `/home/cao/.cache/huggingface/.../models--nvidia--Cosmos-Predict2-2B-Video2World/` | 4.4 GB | `/home/cao/.cache` |
| uv パッケージキャッシュ | `/home/cao/.cache/uv` | ~24 GB | `/home/cao/.cache` |

> `/home/cao/.cache` は `docker_run.sh` が意図的にホストキャッシュ（`/data1/cao/.cache`）をマウントしている
> 再利用用ディレクトリ。マウント外（ホストの他領域）への書き込みは行っていない。

---

## 4. 既知の警告（いずれも無害・対処不要）

- `[robosuite WARNING] No private macro file found!`
  → robosuite **本体**の private macro 未設定の警告。ROBOCASA.md の `setup_macros.py` は **robocasa** 用で別物。
    評価は問題なく完走するため対処不要（必要なら robosuite 同梱の setup スクリプトを実行可）。
- `Could not import robosuite_models` / `mink-based whole-body IK` / `mimicgen environments not imported`
  → 本評価では不要なオプション依存。
- `Skipping key ... _extra_state introduced by TransformerEngine for FP8` (多数)
  → FP8 用の補助 state を読み飛ばすだけ。`_IncompatibleKeys(missing_keys=[], unexpected_keys=[])` で重みは完全一致。
- `resume_download is deprecated` (huggingface_hub) / `SyntaxWarning`(megatron) → 動作に影響なし。

---

## 5. README からの差分まとめ

| # | README/ROBOCASA.md | 本検証で実際に行ったこと | 理由 |
|---|---|---|---|
| 1 | `docker build` / `docker run` | 既存の起動済みコンテナ内で作業 | コンテナは事前構築・稼働済み |
| 2 | `git clone <repo>` | `git clone --depth 1 <repo>` | 取得高速化（任意。フルでも可） |
| 3 | （記載なし） | **`HF_TOKEN` を設定**（gated repo アクセス） | `nvidia/Cosmos-Predict2-2B-Video2World` が gated のため必須 |
| 4 | `--num_trials_per_task 50` | `--num_trials_per_task 2` | 「少量サブセットで評価が走ること」の確認が目的 |

以上により、**Cosmos Policy の RoboCasa 評価パイプラインが本環境で正しく動作することを確認した。**
