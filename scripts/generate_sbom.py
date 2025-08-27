#!/usr/bin/env python3
"""Generate a Software Bill of Materials (SBOM) for the project.

Primary method: Syft (if installed) to produce SPDX JSON & CycloneDX JSON.
Fallback: Pure Python environment introspection (pip metadata) producing a minimal SPDX-like JSON.

Outputs:
  build_artifacts/sbom/spdx-syft.json (if syft available)
  build_artifacts/sbom/cyclonedx-syft.json (if syft available)
  build_artifacts/sbom/minimal-spdx.json (always)
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

SBOM_DIR = Path('build_artifacts/sbom')
SBOM_DIR.mkdir(parents=True, exist_ok=True)


def _run_syft(format_flag: str, output_file: Path) -> bool:
    cmd = ["syft", f"dir:.", f"-o", format_flag]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        output_file.write_bytes(out)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:  # pragma: no cover
        print(f"[WARN] syft failed: {e.output.decode(errors='ignore')[:300]}")
        return False


@dataclass
class MinimalComponent:
    name: str
    version: str
    purl: str | None = None
    license: str | None = None


def _gather_python_packages() -> List[MinimalComponent]:
    comps: List[MinimalComponent] = []
    try:
        import importlib.metadata as md  # Python 3.11+
        for dist in md.distributions():
            name = dist.metadata.get('Name') or dist.metadata.get('Summary') or dist.metadata.get('name')
            version = dist.version
            license_field = dist.metadata.get('License')
            purl = f"pkg:pypi/{(name or dist.metadata['Name']).replace(' ', '-') }@{version}" if name else None
            comps.append(MinimalComponent(name=name or dist.metadata['Name'], version=version, purl=purl, license=license_field))
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to gather python packages: {e}")
    return comps


def _write_minimal_spdx(components: List[MinimalComponent]):
    doc: Dict[str, Any] = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "ai-trading-system-minimal-sbom",
        "creationInfo": {"created": ""},
        "packages": [
            {
                "name": c.name,
                "SPDXID": f"SPDXRef-Package-{i}",
                "versionInfo": c.version,
                **({"downloadLocation": "NOASSERTION"}),
                **({"licenseConcluded": c.license or "NOASSERTION"}),
                **({"externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": c.purl}] if c.purl else {}}),
            }
            for i, c in enumerate(components)
        ],
    }
    out = SBOM_DIR / 'minimal-spdx.json'
    out.write_text(json.dumps(doc, indent=2))
    return out


def main():  # pragma: no cover - orchestration
    print("Generating SBOM(s)...")
    spdx_syft = SBOM_DIR / 'spdx-syft.json'
    cyclonedx_syft = SBOM_DIR / 'cyclonedx-syft.json'
    used_syft = False
    if _run_syft('spdx-json', spdx_syft):
        print(f"[INFO] Syft SPDX written: {spdx_syft}")
        if _run_syft('cyclonedx-json', cyclonedx_syft):
            print(f"[INFO] Syft CycloneDX written: {cyclonedx_syft}")
        used_syft = True
    else:
        print("[WARN] syft not installed; falling back to minimal Python-package SBOM.")
    comps = _gather_python_packages()
    minimal_path = _write_minimal_spdx(comps)
    print(f"[INFO] Minimal SPDX written: {minimal_path}")
    if not used_syft:
        print("[HINT] Install syft: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin")


if __name__ == "__main__":
    main()
