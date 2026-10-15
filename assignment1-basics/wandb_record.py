from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import uuid
import wandb
import dotenv

dotenv.load_dotenv()


@contextmanager
def experiment(task, config, root="."):
    root = Path(root).resolve()
    skipped = {".git", ".venv", "venv", "outputs", "wandb", "data", "target",
               "datasets", "checkpoints", "__pycache__", ".pytest_cache"}
    suffixes = {".py", ".sh", ".ipynb", ".cu", ".c", ".cpp", ".h", ".rs"}
    sources = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skipped)
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root)
            if (path.suffix in suffixes
                or name in {"pyproject.toml", "Cargo.toml", "Cargo.lock", "unicode.bin", "uv.lock", ".python-version"}
                or (rel.parts[0] == "configs" and path.suffix in {".json", ".yaml", ".yml", ".toml"})):
                sources.append(path)
    with wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ.get("WANDB_PROJECT", "lmfs-assignment1"),
        id=uuid.uuid4().hex[:12], resume="never",
        name=f"{os.environ['STUDENT_ID']}-{task}",
        group=os.environ["STUDENT_ID"], job_type=task, config=config,
        settings=wandb.Settings(console="wrap"),
    ) as run:
        out = root / "outputs" / run.id
        out.mkdir(parents=True, exist_ok=True)
        artifact = wandb.Artifact(f"source-{run.id}", type="source-code")
        dump, hashes = [], {}
        for path in sources:
            rel = path.relative_to(root).as_posix()
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            hashes[path] = digest
            frozen = out / "source" / rel
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(content)
            artifact.add_file(str(frozen), name=rel)
            if path.suffix == ".bin":
                block = f"===== BINARY {rel} SHA256={digest} =====\n"
            else:
                lines = "\n".join(f"{i:05d} | {line}" for i, line in
                                  enumerate(content.decode("utf-8-sig").splitlines(), 1))
                block = f"===== SOURCE {rel} SHA256={digest} =====\n{lines}\n"
            print(block, flush=True)
            dump.append(block)
        source_dump = out / "source_dump.txt"
        source_dump.write_text("\n".join(dump), encoding="utf-8")
        artifact.add_file(str(source_dump), name="source_dump.txt")
        run.log_artifact(artifact)
        run.save(str(source_dump), base_path=str(out), policy="now")
        try:
            yield run
        finally:
            changed = [str(path.relative_to(root)) for path, digest in hashes.items()
                       if not path.is_file()
                       or hashlib.sha256(path.read_bytes()).hexdigest() != digest]
            run.summary["source_changed"] = changed
            if changed:
                print("WARNING: 运行中源码发生变化：", changed, flush=True)
