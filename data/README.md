# Data files (not in git)

These are too large for a git repository — `Connectivity_783.parquet` is 100.8 MB, which is
over GitHub's 100 MB per-file hard limit. They are gitignored; fetch them once:

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git /tmp/dbm
cp /tmp/dbm/Connectivity_783.parquet /tmp/dbm/Completeness_783.csv data/
```

FlyWire v783 connectivity and neuron list, from the published analysis of
Shiu, Sterne, Spiller et al., *Nature* (2024). MIT-licensed repo; FlyWire data is CC-BY-NC.

For soma positions and cell typing (`flywire_783_annotations.csv.gz`):

```bash
git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git /tmp/fwann
python3 - <<'PY'
import pandas as pd
cols = ['root_id','pos_x','pos_y','pos_z','soma_x','soma_y','soma_z',
        'super_class','cell_class','cell_type','hemibrain_type','side','top_nt']
a = pd.read_csv('/tmp/fwann/supplemental_files/Supplemental_file1_neuron_annotations.tsv',
                sep='\t', low_memory=False, usecols=cols)
a.to_csv('data/flywire_783_annotations.csv.gz', index=False, compression='gzip')
PY
```

Whole-brain annotation release, Schlegel et al., *Nature* (2024). CC-BY-NC.
