"""Package the local benchmark corpus and application for a Linux host."""

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = [
        "Dockerfile", ".dockerignore", ".env.host.example", "compose.host.yaml",
        "pyproject.toml", "README.md", "LICENSE", "deploy/Caddyfile", "deploy/init.sh",
        "docs/hosting.md", "docs/failure-analysis.md", "eval/README.md",
        "docker-compose.yml", "deploy/package.py",
    ]
    for pattern in (
        "babyinfoquick/**/*.py", "babyinfoquick/**/*.yaml", "configs/*.yaml",
        "templates/*.html", "eval/*.jsonl", "eval/benchmark/*.json",
        "eval/benchmark/*.jsonl", "eval/benchmark/README.md",
    ):
        files.extend(path.relative_to(ROOT).as_posix() for path in sorted(ROOT.glob(pattern)))

    target = ROOT / "dist" / "babyinfoquick-host.tar.gz"
    target.parent.mkdir(exist_ok=True)
    pending = target.with_suffix(".tmp")
    with tarfile.open(pending, "w:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name)
        for strategy in ("section", "fixed"):
            name = f"chunks_{strategy}.jsonl"
            archive.add(ROOT / "data" / "chunks" / name, arcname=f"deploy/seed/{name}")
    pending.replace(target)
    print(target)


if __name__ == "__main__":
    main()
