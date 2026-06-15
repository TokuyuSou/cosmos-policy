#!/usr/bin/env bash
# Cosmos Policy 用 Docker ヘルパー
#   - イメージのビルド
#   - 永続コンテナ(exit しても消えない)の起動 / 入室 / 停止 / 削除
#   - /data1/cao を /workspace にマウント
#
# 使い方:
#   ./docker_run.sh build      # イメージをビルド
#   ./docker_run.sh up         # 永続コンテナを起動 (デタッチ常駐)
#   ./docker_run.sh enter      # コンテナに入る (cao ユーザー)
#   ./docker_run.sh stop       # コンテナを停止 (消さない)
#   ./docker_run.sh start      # 停止したコンテナを再開
#   ./docker_run.sh status     # 状態確認
#   ./docker_run.sh rm         # コンテナを削除 (明示時のみ)
#   ./docker_run.sh logs       # コンテナのログ
set -euo pipefail

# --- 設定 ---------------------------------------------------------------
IMAGE_NAME="cosmos-policy"
CONTAINER_NAME="cosmos-policy"
REPO_DIR="/data1/cao/cosmos-policy"   # Dockerfile があるリポジトリ
MOUNT_SRC="/data1/cao"                # ホスト側: /workspace にマウントする
MOUNT_DST="/workspace"
WORKDIR="/workspace/cosmos-policy"    # 入室時の作業ディレクトリ(リポジトリ直下)
# 全キャッシュ(uv / huggingface / pip / torch 等)を /data1 上に置く。
# ホスト側 CACHE_DIR を コンテナ内 ~/.cache にマウントするので、~/.cache を
# 既定先とするツールは自動的に /data1 を使う。さらに念のため環境変数でも明示する。
CACHE_DIR="/data1/cao/.cache"        # ホスト側キャッシュ置き場 (/data1)
USER_NAME="$(id -un)"
CONTAINER_HOME="/home/${USER_NAME}"  # entrypoint がマッピングするコンテナ内ホーム
CONTAINER_CACHE="${CONTAINER_HOME}/.cache"
# GPU: 全デバイスを可視にする。絞りたい時のみ GPUS='"device=0,1"' で上書き
GPUS="${GPUS:-all}"
# 共有メモリ: 既定は他ユーザーと隔離するため --shm-size を使う(--ipc=host は使わない)。
#   ホストの IPC 名前空間 / /dev/shm を共有しないので、他コンテナの共有メモリに触れない。
#   PyTorch のデータローダ向けに 32g を確保。必要に応じ SHM=16g 等で上書き可。
#   どうしてもホスト IPC を共有したい場合のみ SHM="" USE_IPC_HOST=1 (非推奨)。
SHM="${SHM:-32g}"
USE_IPC_HOST="${USE_IPC_HOST:-0}"
# GPU グラフィクス capability: LIBERO/RoboCasa の MuJoCo オフスクリーン描画は
#   EGL を使うため、graphics/display/video を含む全 capability をコンテナへ公開する。
#   ("all" = compute,utility,graphics,display,video,... を含む)
#   ホストのドライバに整合する graphics ライブラリ(libEGL_nvidia 等)が入っていれば
#   NVIDIA Container Toolkit が自動でマウントする。無ければマウントされないだけで害は無い。
NV_CAPS="${NV_CAPS:-all}"
# MuJoCo / PyOpenGL のレンダリングバックエンド。GPU EGL を使う。
#   (どうしても GPU EGL が使えない時は MUJOCO_GL=osmesa で CPU 描画も可。
#    ただし osmesa はイメージに libosmesa6 が無いと動かない。既定は egl。)
MUJOCO_GL="${MUJOCO_GL:-egl}"
# -----------------------------------------------------------------------

cmd="${1:-help}"

build() {
  mkdir -p "${CACHE_DIR}"
  echo ">> イメージ ${IMAGE_NAME} をビルドします (context: ${REPO_DIR}/docker)"
  DOCKER_BUILDKIT=1 docker build -t "${IMAGE_NAME}" "${REPO_DIR}/docker"
}

# 共有メモリ系のオプションを組み立て
shm_opts() {
  if [[ -n "${SHM}" ]]; then
    printf -- '--shm-size %s' "${SHM}"
  elif [[ "${USE_IPC_HOST}" == "1" ]]; then
    printf -- '--ipc=host'
  fi
}

up() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo ">> コンテナ ${CONTAINER_NAME} は既に存在します。"
    echo "   入室: $0 enter / 再開: $0 start / 作り直し: $0 rm && $0 up"
    return 0
  fi
  mkdir -p "${CACHE_DIR}"
  echo ">> 永続コンテナ ${CONTAINER_NAME} を起動します (デタッチ常駐)"
  # entrypoint(cosmos-policy-entrypoint) を上書きしないことで、
  #   コンテナ内ユーザーを host cao (uid:gid $(id -u):$(id -g)) にマッピングし
  #   gosu で降格する => 常駐プロセスも作業も cao(非 root)で動き、
  #   /workspace 上の生成物が cao 所有になる(root 汚染なし)。
  # --security-opt no-new-privileges: setuid 等での root 昇格を禁止(保険)。
  #   gosu による権限「降格」はブロックされないので動作に影響しない。
  # CMD を sleep infinity にして常駐させ、作業は exec で入る。
  # --rm を付けないので exit / stop しても消えない。
  # キャッシュ環境変数は docker run 時に設定する。こうすると enter(exec) でも
  # VSCode の Attach でも継承され、uv / hf / pip / torch が全て /data1 を使う。
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --security-opt no-new-privileges \
    -e HOST_USER_ID="$(id -u)" \
    -e HOST_GROUP_ID="$(id -g)" \
    -e HOST_USER_NAME="${USER_NAME}" \
    -e XDG_CACHE_HOME="${CONTAINER_CACHE}" \
    -e HF_HOME="${CONTAINER_CACHE}/huggingface" \
    -e HUGGINGFACE_HUB_CACHE="${CONTAINER_CACHE}/huggingface/hub" \
    -e UV_CACHE_DIR="${CONTAINER_CACHE}/uv" \
    -e PIP_CACHE_DIR="${CONTAINER_CACHE}/pip" \
    -e TORCH_HOME="${CONTAINER_CACHE}/torch" \
    -e NVIDIA_DRIVER_CAPABILITIES="${NV_CAPS}" \
    -e MUJOCO_GL="${MUJOCO_GL}" \
    -e PYOPENGL_PLATFORM="${MUJOCO_GL}" \
    -v "${CACHE_DIR}:${CONTAINER_CACHE}" \
    -v "${MOUNT_SRC}:${MOUNT_DST}" \
    --gpus "${GPUS}" \
    $(shm_opts) \
    -w "${WORKDIR}" \
    "${IMAGE_NAME}" sleep infinity
  echo ">> 起動しました。入室するには: $0 enter"
}

enter() {
  # 常に非 root の作業ユーザー(cao)で入る。docker exec は entrypoint を経由しない
  # ため、ユーザー / HOME を明示する。root では入らない(安全側の既定)。
  echo ">> user=$(id -un) (非 root) で ${CONTAINER_NAME} に入ります"
  docker exec -it \
    -u "${USER_NAME}" \
    -w "${WORKDIR}" \
    -e HOME="${CONTAINER_HOME}" \
    "${CONTAINER_NAME}" bash
}

stop()   { docker stop "${CONTAINER_NAME}"; }
start()  { docker start "${CONTAINER_NAME}"; echo ">> 入室: $0 enter"; }
status() { docker ps -a --filter "name=^/${CONTAINER_NAME}$"; }
logs()   { docker logs "${CONTAINER_NAME}"; }
remove() {
  docker rm -f "${CONTAINER_NAME}"
  echo ">> コンテナ ${CONTAINER_NAME} を削除しました。"
}

case "${cmd}" in
  build)  build ;;
  up)     up ;;
  enter)  enter ;;
  stop)   stop ;;
  start)  start ;;
  status) status ;;
  logs)   logs ;;
  rm)     remove ;;
  *)
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
