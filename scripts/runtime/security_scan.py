"""Runtime copy (Phase A) of security scan script.
Renamed from security_scan_local.py for clearer semantics.
Original retained until Phase B.
"""

import subprocess, shutil, sys

def have(tool: str) -> bool:
    return shutil.which(tool) is not None

def run(cmd):
    try:
        return subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)} (exit {e.returncode})", file=sys.stderr)
        return None

def main():
    # Prefer trivy for container context, fallback grype
    image = sys.argv[1] if len(sys.argv) > 1 else None
    if image and have('trivy'):
        print(f"Running trivy image scan on {image}")
        run(['trivy', 'image', '--quiet', '--severity', 'HIGH,CRITICAL', image])
    elif image and have('grype'):
        print(f"Running grype scan on {image}")
        run(['grype', image])
    else:
        print("No container image supplied or scanners missing; skipping")

if __name__ == '__main__':
    main()
