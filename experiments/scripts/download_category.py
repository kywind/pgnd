#!/usr/bin/env python3
"""Download, verify, and extract one PGND dataset category."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


CATEGORIES = ("box", "bread", "cloth", "paperbag", "rope", "sloth")
DEFAULT_REVISION = "release-20260831"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path.strip()] = digest
    return checksums


def checked_output_path(root: Path, member_name: str) -> Path:
    root = root.resolve()
    path = (root / member_name).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"archive path escapes output directory: {member_name}")
    return path


def extract_archive(archive_path: Path, output: Path) -> tuple[int, int]:
    linked = 0
    copied = 0
    with tarfile.open(archive_path) as archive:
        hardlink_members = []

        def regular_members():
            for member in archive:
                if member.islnk():
                    hardlink_members.append(member)
                else:
                    yield member

        archive.extractall(output, members=regular_members(), filter="data")
        for member in hardlink_members:
            destination = checked_output_path(output, member.name)
            target = checked_output_path(output, member.linkname)
            if not target.is_file():
                raise RuntimeError(
                    f"hardlink target was not extracted: {member.linkname}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            try:
                os.link(target, destination)
                linked += 1
            except OSError:
                shutil.copy2(target, destination)
                copied += 1
    return linked, copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=CATEGORIES)
    parser.add_argument("--repo-id", default="kaifz/pgnd-dataset")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path.cwd())
    parser.add_argument("--download-dir", type=Path, default=Path("pgnd-download"))
    parser.add_argument("--without-assets", action="store_true")
    args = parser.parse_args()

    patterns = [
        "README.md",
        "DATASET_MANIFEST.json",
        "SHA256SUMS",
        f"data/{args.category}/*.tar",
    ]
    if not args.without_assets:
        patterns.append("assets/*.tar")

    download_root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=patterns,
            local_dir=args.download_dir,
        )
    )
    archives = sorted((download_root / "data" / args.category).glob("*.tar"))
    if not args.without_assets:
        archives.extend(sorted((download_root / "assets").glob("*.tar")))
    if not archives:
        raise RuntimeError(f"no archives downloaded for {args.category}")

    checksums = parse_checksums(download_root / "SHA256SUMS")
    for archive in archives:
        relative_path = archive.relative_to(download_root).as_posix()
        expected = checksums.get(relative_path)
        if expected is None:
            raise RuntimeError(f"missing checksum for {relative_path}")
        actual = sha256(archive)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for {relative_path}: {actual} != {expected}"
            )
        print(f"verified {relative_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"extracting {archive} -> {args.output}")
        linked, copied = extract_archive(archive, args.output)
        if linked:
            print(f"created {linked} deduplicated hardlinks")
        if copied:
            print(f"materialized {copied} copies (hardlinks unavailable)")


if __name__ == "__main__":
    main()
