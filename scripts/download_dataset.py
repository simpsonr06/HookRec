from pathlib import Path
import sys
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hookrec.data import extract_archive
URL='https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
ARCHIVE=Path('data/raw/hetrec2011-lastfm-2k.zip'); DEST=Path('data/raw/lastfm-2k')
def main():
    if all((DEST/n).is_file() for n in ('user_artists.dat','artists.dat','user_taggedartists.dat','tags.dat')):
        print(f'Dataset already available: {DEST}'); return
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True); print(f'Downloading {URL}')
    with requests.get(URL,stream=True,timeout=(20,120)) as response:
        response.raise_for_status()
        with ARCHIVE.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk: f.write(chunk)
    extract_archive(ARCHIVE,DEST); print(f'Extracted dataset to {DEST}. Review DATASET.md usage terms.')
if __name__=='__main__': main()
