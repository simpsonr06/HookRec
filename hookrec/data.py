from pathlib import Path
import zipfile
import pandas as pd
FILES={'user_artists.dat','artists.dat','user_taggedartists.dat','tags.dat'}

def extract_archive(archive_path, destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        for member in z.infolist():
            name=Path(member.filename).name
            if name in FILES and not member.is_dir():
                with z.open(member) as src, (destination/name).open('wb') as dst: dst.write(src.read())
    missing=FILES-{p.name for p in destination.iterdir() if p.is_file()}
    if missing: raise ValueError(f'Archive missing files: {sorted(missing)}')
    return destination

def load_dataset(data_dir):
    d=Path(data_dir); missing=FILES-{p.name for p in d.iterdir() if p.is_file()}
    if missing: raise FileNotFoundError(f'Missing {sorted(missing)} in {d}; run python scripts/download_dataset.py')
    inter=pd.read_csv(d/'user_artists.dat',sep='\t', encoding='latin-1').rename(columns={'userID':'user_id','artistID':'artist_id'})
    artists=pd.read_csv(d/'artists.dat',sep='\t', encoding='latin-1').rename(columns={'id':'artist_id','name':'artist_name'})[['artist_id','artist_name']]
    user_tags=pd.read_csv(d/'user_taggedartists.dat',sep='\t', encoding='latin-1').rename(columns={'artistID':'artist_id','tagID':'tag_id'})
    tags=pd.read_csv(d/'tags.dat',sep='\t', encoding='latin-1').rename(columns={'tagID':'tag_id','tagValue':'tag'})
    tag_rows=user_tags.merge(tags[['tag_id','tag']],on='tag_id',how='inner')[['artist_id','tag']].drop_duplicates()
    return inter,artists,tag_rows
