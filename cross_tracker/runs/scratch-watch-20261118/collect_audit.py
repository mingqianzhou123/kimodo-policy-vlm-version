"""Export a read-only, validated snapshot of completed evaluation receipts."""
from pathlib import Path
import argparse
import hashlib
import json
import time

from summarize_final import METHODS, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    batch = json.loads((args.run / 'batch.json').read_text())
    manifests = {method: json.loads(path.read_text()) for method in METHODS
                 if (path := args.run / f'{method}-manifest.json').exists()}
    reports, hashes = {}, {}
    for item in batch['completed']:
        path = Path(item['command'][2]).parents[1] / 'runs' / item['run_id'] / 'run.json'
        data = path.read_bytes()
        reports[item['run_id']] = json.loads(data)
        hashes[item['run_id']] = hashlib.sha256(data).hexdigest()
    result = dict(checked_unix=time.time(), run=str(args.run), batch=batch,
                  manifests=manifests, reports=reports, receipt_sha256=hashes,
                  summary=summarize(batch, reports, manifests, require_complete=False))
    for name in ['prepare', 'source-files', 'shared-eval-assets', 'protocol']:
        result[name] = json.loads((args.run / f'{name}.json').read_text())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print('AUDIT', args.output, args.output.stat().st_size,
          hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)
    for cell in result['summary']['cells']:
        print(cell['method'], cell['controller'], cell['successes'], '/', cell['completed'])


if __name__ == '__main__':
    main()
