#!/usr/bin/env bash
set -euo pipefail

# 一键编译 LaTeX 论文（Tectonic）
# 用法:
#   ./compile.sh         编译 main.tex（英文版）
#   ./compile.sh cn      编译 main_cn.tex（中文版）
#   ./compile.sh clean   清理辅助文件

TEC="$HOME/.local/bin/tectonic"
MAIN="main"
CN="main_cn"

compile() {
  local name="$1"
  echo "===== 编译 ${name}.tex ====="

  # tectonic -X 自动下载缺失宏包，自动 bibtex
  # -k 保留中间文件以加速后续编译
  "$TEC" -X compile -k --reruns 3 "${name}.tex"

  echo "===== 完成: ${name}.pdf ====="
}

clean() {
  echo "清理辅助文件..."
  rm -f -- *.aux *.bbl *.blg *.log *.out *.synctex.gz *.fls *.fdb_latexmk *.run.xml *.xdv *.nav *.snm *.toc *.vrb *.soc *.listing *.loa *.lof *.lot *.tdo
}

case "${1:-en}" in
  cn)   compile "$CN" ;;
  en)   compile "$MAIN" ;;
  clean) clean ;;
  *)
    echo "用法: ./compile.sh [en|cn|clean]"
    exit 1
    ;;
esac
