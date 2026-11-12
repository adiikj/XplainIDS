# Datasets

The datasets are not in this repository. Download them and place the files exactly as below; `data.py` reads these paths.

```
data/nsl_kdd/KDDTrain+.txt
data/nsl_kdd/KDDTest+.txt
data/unsw_nb15/train.csv
data/unsw_nb15/test.csv
```

- **NSL-KDD**: https://www.unb.ca/cic/datasets/nsl.html (the same files are mirrored at github.com/defcom17/NSL_KDD).
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset, the official training and testing partitions
  (mirrored at huggingface.co/datasets/Mireu-Lab/UNSW-NB15). In that mirror the two names are swapped: `test.csv` holds the
  175,341-row training partition and `train.csv` the 82,332-row testing partition. Keep the mirror's names; `data.py` handles the swap.

Please cite the dataset authors (references in `RESULTS.md` and the papers) when you use them.
