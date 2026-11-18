"""Verify and back up the CCT final inside CCI, independent of public SSH access."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    with (HERE / 'cct-backup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                process = subprocess.run([sys.executable, str(HERE / 'verify_cct_final.py')],
                                         capture_output=True, text=True, timeout=300)
                if process.returncode:
                    raise RuntimeError(process.stderr[-1500:])
                report = json.loads(process.stdout)
                temporary = HERE / 'cct-server-latest.tmp'
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(HERE / 'cct-server-latest.json')
                print(time.strftime('%F %T'), report['status'], report['latest_metric']['step'], flush=True)
                if report['status'] == 'verified_final':
                    return
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                print(time.strftime('%F %T'), 'BACKUP_CHECK_RETRY', repr(error), flush=True)
            time.sleep(300)


if __name__ == '__main__':
    main()
