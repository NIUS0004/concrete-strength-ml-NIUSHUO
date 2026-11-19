"""Download the original UCI archive and retain its attribution."""
import hashlib
import io
import json
import urllib.request
import zipfile
import pandas as pd
from common import ROOT, FEATURES, TARGET

URL = 'https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip'

def main():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    archive = folder / 'uci_concrete.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith('Concrete_Data.xls'))
        df = pd.read_excel(io.BytesIO(z.read(name)), engine='xlrd')
        for n in z.namelist():
            if n.lower().endswith('.txt'):
                (folder / 'original_readme.txt').write_bytes(z.read(n))
    assert df.shape == (1030, 9), f'Unexpected UCI schema: {df.shape}'
    df.columns = FEATURES + [TARGET]
    df.to_csv(folder / 'concrete.csv', index=False)
    (folder / 'provenance.json').write_text(json.dumps({
        'source_url': URL, 'doi': '10.24432/C5PK67', 'author': 'I-Cheng Yeh',
        'license': 'CC BY 4.0', 'archive_sha256': hashlib.sha256(raw).hexdigest(),
        'transformation': 'Original row order and values retained; columns renamed; exported as CSV.'
    }, indent=2), encoding='utf-8')
    print(f'Saved {len(df)} rows to data/concrete.csv')

if __name__ == '__main__':
    main()
