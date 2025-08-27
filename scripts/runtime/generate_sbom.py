"""Runtime copy of SBOM generation script (Phase A duplication).

Once validated, the original scripts/generate_sbom.py can be removed.
"""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import datetime

OUTPUT_DIR = Path("sbom")
OUTPUT_DIR.mkdir(exist_ok=True)

timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
spdx_path = OUTPUT_DIR / f"sbom-{timestamp}.spdx.json"
cyclonedx_path = OUTPUT_DIR / f"sbom-{timestamp}.cyclonedx.json"

def have_tool(tool: str) -> bool:
    return shutil.which(tool) is not None

def run(cmd):
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return None

def generate_with_syft():
    # Try both SPDX and CycloneDX JSON outputs
    rc1 = run(["syft", "packages:", "-o", f"spdx-json={spdx_path}"])
    rc2 = run(["syft", "packages:", "-o", f"cyclonedx-json={cyclonedx_path}"])
    return rc1 is not None or rc2 is not None

def fallback_minimal():
    data = {
        "bomFormat": "SPDX",
        "spdxVersion": "SPDX-2.3",
        "name": "ai-trading-system-minimal",
        "creationInfo": {
            "created": timestamp,
            "creators": ["Tool: minimal-fallback"]
        },
        "packages": []
    }
    spdx_path.write_text(json.dumps(data, indent=2))
    # simple CycloneDX-like structure
    cdx = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"timestamp": timestamp},
        "components": []
    }
    cyclonedx_path.write_text(json.dumps(cdx, indent=2))
    print("Fallback minimal SBOMs written")

def main():
    if have_tool("syft"):
        if generate_with_syft():
            print(f"SBOM generated: {spdx_path if spdx_path.exists() else ''} {cyclonedx_path if cyclonedx_path.exists() else ''}")
            return
        else:
            print("Syft present but generation failed, using fallback", file=sys.stderr)
    else:
        print("Syft not found, using fallback minimal SBOM", file=sys.stderr)
    fallback_minimal()

if __name__ == "__main__":
    main()