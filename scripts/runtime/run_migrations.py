"""Runtime copy (Phase A) of migration runner.
Original retained until Phase B.
"""

import os, subprocess, sys

ALEMBIC_CFG = os.environ.get('ALEMBIC_CONFIG', 'alembic.ini')

def main():
    cmd = ['alembic', '-c', ALEMBIC_CFG, 'upgrade', 'head']
    print('Executing:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Migration failed (exit {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
