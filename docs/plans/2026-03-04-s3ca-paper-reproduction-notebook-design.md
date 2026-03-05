# Design: S3CA Paper Reproduction Notebook (Example 10)

Date: 2026-03-04

## Goal

Create `examples/10_s3ca_paper_reproduction.ipynb` as a Jupyter notebook companion to
`examples/10_s3ca_paper_reproduction.py`, following the established 1:1 translation
pattern used by all other examples in this project.

## Pattern

Every `.py` example has a matching `.ipynb`. The notebook:
- Replaces `matplotlib.use("Agg")` with `%matplotlib inline`
- Splits the script at section comment boundaries into markdown + code cell pairs
- Keeps `savefig()` calls so outputs land in `examples/outputs/`
- Adds a Summary markdown cell at the end

## Cell Structure

| Cell | Type     | Content |
|------|----------|---------|
| 0    | Markdown | Title, level, paper reference, learning objectives |
| 1    | Code     | Imports (`%matplotlib inline`) |
| 2    | Markdown | `## 1. Generate DSSS-BPSK Signal` |
| 3    | Code     | Signal generation + AWGN noise |
| 4    | Markdown | `## 2. Compute SCD` |
| 5    | Code     | n_alpha calc, reference SCD (kappa=n_alpha), S3CA (kappa=80) |
| 6    | Markdown | `## 3. Alpha Profile` |
| 7    | Code     | Alpha axis computation, max profiles |
| 8    | Markdown | `## 4. Plot Figure 3 Reproduction` |
| 9    | Code     | Full 6-panel figure + kappa sweep loop |
| 10   | Markdown | Summary |

## Notes

- The kappa sweep (panel f) stays in the same cell as the figure — it is tightly coupled to the plot.
- No additions beyond what is in the `.py` file.
