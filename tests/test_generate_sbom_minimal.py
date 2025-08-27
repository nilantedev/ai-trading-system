from pathlib import Path
import json
import subprocess
import sys


def test_minimal_sbom_generation():
    # Run the script in a temporary working directory copying pyproject only to limit size
    project_root = Path(__file__).resolve().parents[1]
    sbom_script = project_root / 'scripts' / 'generate_sbom.py'
    # Execute script; syft likely absent -> fallback path exercised
    result = subprocess.run([sys.executable, str(sbom_script)], cwd=project_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    sbom_file = project_root / 'build_artifacts' / 'sbom' / 'minimal-spdx.json'
    assert sbom_file.exists(), "minimal SBOM file should be created"
    data = json.loads(sbom_file.read_text())
    assert data.get('spdxVersion') == 'SPDX-2.3'
    assert 'packages' in data and isinstance(data['packages'], list)
