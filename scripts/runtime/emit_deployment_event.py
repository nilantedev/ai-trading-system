"""Runtime copy (Phase A) of deployment event emission script.
Original retained at scripts/emit_deployment_event.py until Phase B pruning.
"""

import json, os, datetime, hashlib, platform, socket, getpass, uuid, pathlib

OUT_DIR = pathlib.Path("deployment_events")
OUT_DIR.mkdir(exist_ok=True)

def hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest_path = pathlib.Path('deployment.manifest')
    manifest_hash = hash_file(manifest_path) if manifest_path.exists() else None
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "commit": os.getenv('GIT_COMMIT_SHA'),
        "deployment_environment": os.getenv('DEPLOY_ENV', 'unknown'),
        "manifest_hash": manifest_hash,
    }
    out_file = OUT_DIR / f"deployment_event_{event['event_id']}.json"
    out_file.write_text(json.dumps(event, indent=2))
    print(f"Deployment event written to {out_file}")

if __name__ == "__main__":
    main()
