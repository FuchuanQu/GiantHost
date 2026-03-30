#!/usr/bin/env bash
set -euo pipefail

# GiantHost database download helper.

DATABASE_URL="https://github.com/FuchuanQu/GiantHost/releases/download/0.1.0/gianthost_db.zip"
DATABASE_ARCHIVE_NAME="gianthost_db.zip"
DATABASE_MD5="25ab1f8f7c34feba6631aa0849bc9694"

TARGET_DIR="${1:-./db}"

if [[ -z "${DATABASE_URL}" ]]; then
  echo "[ERROR] DATABASE_URL is empty."
  echo "Edit scripts/download_database.sh and set DATABASE_URL before using this script."
  exit 1
fi

mkdir -p "${TARGET_DIR}"
ARCHIVE_PATH="${TARGET_DIR}/${DATABASE_ARCHIVE_NAME}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: ${cmd}"
    exit 1
  fi
}

verify_md5_if_needed() {
  if [[ -z "${DATABASE_MD5}" ]]; then
    echo "[INFO] DATABASE_MD5 is empty. Skipping checksum verification."
    return 0
  fi

  local actual=""
  if command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "${ARCHIVE_PATH}" | awk '{print $1}')"
  elif command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "${ARCHIVE_PATH}")"
  else
    echo "[ERROR] DATABASE_MD5 is set but neither md5sum nor md5 is available."
    exit 1
  fi

  if [[ "${actual}" != "${DATABASE_MD5}" ]]; then
    echo "[ERROR] MD5 mismatch."
    echo "        expected: ${DATABASE_MD5}"
    echo "        actual:   ${actual}"
    exit 1
  fi

  echo "[INFO] MD5 checksum verified."
}

extract_archive() {
  local file="$1"
  local dest="$2"
  local lower
  lower="$(echo "${file}" | tr '[:upper:]' '[:lower:]')"

  case "${lower}" in
    *.tar.gz|*.tgz|*.tar)
      require_cmd tar
      tar -xf "${file}" -C "${dest}"
      return 0
      ;;
    *.tar.bz2|*.tbz2)
      require_cmd tar
      tar -xjf "${file}" -C "${dest}"
      return 0
      ;;
    *.tar.xz|*.txz)
      require_cmd tar
      tar -xJf "${file}" -C "${dest}"
      return 0
      ;;
    *.zip)
      require_cmd unzip
      unzip -o "${file}" -d "${dest}"
      return 0
      ;;
  esac

  # Fallback: try tar first, then zip.
  if command -v tar >/dev/null 2>&1 && tar -tf "${file}" >/dev/null 2>&1; then
    tar -xf "${file}" -C "${dest}"
    return 0
  fi

  if command -v unzip >/dev/null 2>&1 && unzip -tq "${file}" >/dev/null 2>&1; then
    unzip -o "${file}" -d "${dest}"
    return 0
  fi

  echo "[ERROR] Unsupported or corrupted archive format: ${file}"
  exit 1
}

echo "[INFO] Downloading database archive..."
if command -v curl >/dev/null 2>&1; then
  curl -L "${DATABASE_URL}" -o "${ARCHIVE_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget "${DATABASE_URL}" -O "${ARCHIVE_PATH}"
else
  echo "[ERROR] Neither curl nor wget is available."
  exit 1
fi

verify_md5_if_needed

echo "[INFO] Extracting database archive to ${TARGET_DIR} ..."
extract_archive "${ARCHIVE_PATH}" "${TARGET_DIR}"

echo "[INFO] Done. Please check that the db folder contains:"
echo "       - ncldv_complete.fasta"
echo "       - taxonomy.csv"
echo "       - gvog.complete.hmm"
echo "       - model/"
