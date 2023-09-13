# Learning minimal volume uncertainty ellipsoids

This repository includes the code attached to the article...

## How to run

1. Run compare.py:
```
python compare.py -database [[DB_NAME:str]] -q [[Q:float]] -seed [[SEED:int]]
```
Where the database name ``[[DB_NAME:str]]`` is one of the database folder names in the datasets folder, q ``[[Q:float]]`` is the coverage constraint (for example 0.9), and seed ``[[SEED:int]]`` is the number of the run to reproduce same results.

2. Run merge.py:
```
python merge.py -file [[DB_NAME:str]] -q [[Q:float]] -total [[TOTAL:int]]
```
Where database name and q are the same, and total ``[[TOTAL:int]]`` is how many experiments exist.
