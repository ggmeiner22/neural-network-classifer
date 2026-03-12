# neural-network-classifer


## Compalation and Execution
```
chmod -R u+w .
chmod +x *.sh
chmod +x scripts/*.sh
chmod +x src/*.cpp
```
### Run All
```
./run_all.sh
```
### Run Individually
#### Build
```
make
```
#### Identity
```
./scripts/run_identity.sh
```
#### Tennis
```
./scripts/run_tennis.sh
```
#### Iris
```
./scripts/run_iris.sh
```
#### IrisNoisy
```
./scripts/run_irisNoisy.sh
```
#### Clean object files
```
make clean
```

## File Structure
```
neural-network-classifer/
├── data/
│   ├── identity-attr.txt
│   ├── identity-train.txt
│   ├── iris-attr.txt
│   ├── iris-test.txt
│   ├── iris-train.txt
│   ├── tennis-attr.txt
│   ├── tennis-test.txt
│   └── tennis-train.txt
├── include/
│   ├── AttrParser.h
│   ├── Dataset.h
│   ├── MLP.h
│   └── Util.h
├── scripts/
│   ├── grid_search.sh
│   ├── make_plot.sh
│   ├── run_identity.sh
│   ├── run_iris.sh
│   ├── run_irisNoisy.sh
│   └── run_tennis.sh
├── src/
│   ├── AttrParser.cpp
│   ├── Dataset.cpp
│   ├── MLP.cpp
│   ├── Util.cpp
│   └── main.cpp
|
├── Makefile
└── run_all.sh
```
## File Overview

