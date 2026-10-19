"""Download only four measurement tables per benchmark at one immutable revision."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata, get_token

ROOT=Path(__file__).resolve().parents[1]
REPO='aims-foundations/measurement-db'
BENCHMARKS=['matharena','mmdocrag','multi_swebench','real_webagents','researchcodebench','swe_rebench']

def main():
    out=ROOT/'data/measurement-db'
    parser=argparse.ArgumentParser();parser.add_argument('--revision');args=parser.parse_args()
    saved=out/'coldcast_download_manifest.json'
    revision=args.revision or (json.loads(saved.read_text())['revision'] if saved.exists() else
        get_hf_file_metadata(hf_hub_url(REPO,'README.md',repo_type='dataset'),token=get_token(),timeout=20).commit_hash)
    print('Pinned revision:',revision,flush=True)
    names=['README.md']+[f'{b}/{name}' for b in BENCHMARKS for name in
        ['metadata.yaml','characterization.yaml','formatted_tables/benchmarks.parquet',
         'formatted_tables/subjects.parquet','formatted_tables/items.parquet','formatted_tables/responses.parquet']]
    def inspect(name):
        m=get_hf_file_metadata(hf_hub_url(REPO,name,repo_type='dataset',revision=revision),token=get_token(),timeout=20)
        return dict(path=name,size=m.size)
    with ThreadPoolExecutor(max_workers=4) as pool: inventory=list(pool.map(inspect,names))
    total=sum(x['size'] or 0 for x in inventory)
    print(f'Selected {len(names)} files; {total/1e6:.1f} MB; no traces, images, or raw archives',flush=True)
    if total>1_000_000_000: raise RuntimeError('Selected tables exceed the 1 GB download limit')
    def fetch(entry):
        p=Path(hf_hub_download(REPO,entry['path'],repo_type='dataset',revision=revision,local_dir=out))
        entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        print('Downloaded',entry['path'],flush=True)
        return entry
    with ThreadPoolExecutor(max_workers=4) as pool: files=list(pool.map(fetch,inventory))
    (out/'coldcast_download_manifest.json').write_text(json.dumps(dict(repository=REPO,revision=revision,total_bytes=total,files=files),indent=2))

if __name__=='__main__':main()
