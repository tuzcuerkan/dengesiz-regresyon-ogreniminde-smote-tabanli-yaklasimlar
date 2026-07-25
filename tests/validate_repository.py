from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import py_compile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
IGNORED_PARTS = {".git", ".venv", ".runs", "__pycache__", ".pytest_cache"}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> set[str]:
    files: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        files.add(relative.as_posix())
    return files


def validate_manifest() -> None:
    if not MANIFEST.is_file():
        fail("SHA256SUMS bulunamadı.")
    rows: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split("  ", 1)
        if path in rows:
            fail(f"Checksum envanterinde yinelenen yol: {path}")
        rows[path] = digest

    actual_files = repository_files()
    manifest_files = set(rows)
    missing = sorted(manifest_files - actual_files)
    unlisted = sorted(actual_files - manifest_files)
    if missing:
        fail(f"Checksum envanterindeki dosyalar bulunamadı: {missing}")
    if unlisted:
        fail(f"Checksum envanterine alınmamış dosyalar var: {unlisted}")

    for path, expected in rows.items():
        actual = sha256(ROOT / path)
        if actual != expected:
            fail(f"Checksum uyuşmazlığı: {path}")


def validate_privacy() -> None:
    forbidden_extensions = {
        ".sav", ".dta", ".rds", ".parquet", ".feather", ".xlsx", ".xls", ".zip"
    }
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in forbidden_extensions:
            fail(f"Kısıtlı veri veya arşiv dosyası bulundu: {relative}")
        lowered = path.name.lower()
        if "tgss2024" in lowered or "tgss_2024" in lowered:
            fail(f"TGSS ham veri adı taşıyan dosya bulundu: {relative}")

    raw_files = [
        path for path in (ROOT / "data" / "raw").rglob("*")
        if path.is_file() and path.name != "README.md"
    ]
    if raw_files:
        fail(f"data/raw altında izlenen dosya bulundu: {raw_files}")

    forbidden_aggregate = ROOT / "data" / "aggregates" / "tgss_income_category_frequencies.csv"
    if forbidden_aggregate.exists():
        fail("Küçük hücre içeren TGSS gelir sıklığı dosyası bulundu.")


def validate_public_structure() -> None:
    required_directories = {
        ".github",
        "data",
        "docs",
        "environment",
        "outputs",
        "scripts",
        "src",
        "tests",
        "validation",
    }
    actual_directories = {
        path.name for path in ROOT.iterdir()
        if path.is_dir() and path.name not in IGNORED_PARTS
    }
    if actual_directories != required_directories:
        fail(
            "Kök dizin yapısı kanonik envanterle uyuşmuyor: "
            f"{sorted(actual_directories)}"
        )

    source_packages = {
        path.name for path in (ROOT / "src").iterdir()
        if path.is_dir() and path.name not in IGNORED_PARTS
    }
    if source_packages != {"imbalanced_regression"}:
        fail(f"Beklenmeyen kaynak paketleri bulundu: {sorted(source_packages)}")

    expected_scripts = {
        "run_independent_reproduction.py",
        "run_main_and_data_sensitivity.py",
        "run_protocol_sensitivity_10fold.py",
    }
    actual_scripts = {
        path.name for path in (ROOT / "scripts").glob("*.py")
        if path.is_file()
    }
    if actual_scripts != expected_scripts:
        fail(f"Çalıştırma betiği envanteri uyuşmuyor: {sorted(actual_scripts)}")


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_results() -> None:
    expected_counts = {
        "outputs/results/main_grouped_repeated5fold_fold_results.csv": 400,
        "outputs/results/grouped_10fold_fold_results.csv": 400,
        "outputs/results/independent_reproduction_fold_results.csv": 400,
    }
    for relative, expected in expected_counts.items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"Sonuç dosyası bulunamadı: {relative}")
        actual = count_csv_rows(path)
        if actual != expected:
            fail(f"{relative}: {actual} satır bulundu, beklenen {expected}.")

    experiment = json.loads(
        (ROOT / "validation" / "experiment_validation.json").read_text(encoding="utf-8")
    )
    reproduction = json.loads(
        (ROOT / "validation" / "reproduction_validation.json").read_text(encoding="utf-8")
    )
    if experiment.get("status") != "PASS" or experiment.get("total_evaluations") != 1450:
        fail("Deney doğrulama kaydı başarısız veya eksik.")
    if reproduction.get("status") != "PASS":
        fail("Bağımsız yeniden üretim kaydı başarısız.")
    if reproduction.get("winner_agreement") != {"same": 88, "total": 88}:
        fail("Model birinciliği uyumu 88/88 değil.")
    if reproduction.get("full_rank_agreement") != {"same": 88, "total": 88}:
        fail("Tam sıralama uyumu 88/88 değil.")

    if len(list((ROOT / "outputs" / "tables").rglob("*.csv"))) != 10:
        fail("Beklenen 10 tez tablo kaynağı bulunamadı.")
    if len(list((ROOT / "outputs" / "figures").rglob("*.png"))) != 9:
        fail("Beklenen 9 şekil dosyası bulunamadı.")


def validate_code_and_metadata() -> None:
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        py_compile.compile(str(path), doraise=True)

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "0009-0000-6595-1171" not in citation:
        fail("CITATION.cff içinde doğru ORCID bulunamadı.")

    required_docs = [
        "docs/experiment_protocol.md",
        "docs/methodology.md",
        "docs/data_access_and_privacy.md",
        "docs/reproducibility.md",
        "docs/thesis_outputs.md",
    ]
    for relative in required_docs:
        if not (ROOT / relative).is_file():
            fail(f"Belge bulunamadı: {relative}")


def main() -> None:
    validate_manifest()
    validate_privacy()
    validate_public_structure()
    validate_results()
    validate_code_and_metadata()
    print(
        "PASS: dosya bütünlüğü, TGSS veri güvenliği, kanonik OİHA "
        "terminolojisi ve 1.450 değerlendirmelik araştırma paketi doğrulandı."
    )


if __name__ == "__main__":
    main()
