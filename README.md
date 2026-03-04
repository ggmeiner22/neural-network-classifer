# neural-network-classifer

Make executable:

chmod +x scripts/*.sh

TODO:
stopping criterion (e.g. number of iterations, convergence, or low enough error)

cd /home/ggmeiner22/projects/ml/neural-network-classifer

# tennis
./scripts/grid_search.sh tennis \
    data/tennis-attr.txt data/tennis-train.txt data/tennis-test.txt

# iris
./scripts/grid_search.sh iris \
    data/iris-attr.txt data/iris-train.txt data/iris-test.txt

# iris_noisy (same files, just mode differs)
./scripts/grid_search.sh iris_noisy \
    data/iris-attr.txt data/iris-train.txt data/iris-test.txt

By default the search script now varies the number of epochs (see the
`epochs` array at the top of `scripts/grid_search.sh`); adjust that list
if you want to tune training duration as well.

./scripts/grid_search.sh identity \
    data/identity-attr.txt data/identity-train.txt
