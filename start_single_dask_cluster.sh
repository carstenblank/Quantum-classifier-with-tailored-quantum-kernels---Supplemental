#!/usr/bin/env bash

python3 -m venv ./venv
. ./venv/bin/activate
pip install wheel --upgrade


if [[ ! -d "distributed" ]]; then
  git clone https://github.com/dask/distributed.git
fi
cd distributed || exit 1
git pull
pip install .
cd ..

if [[ ! -d "dask" ]]; then
  git clone https://github.com/dask/dask.git
fi
cd dask || exit 1
git pull
pip install ".[complete]" || exit 1
cd ..

pip install -r ./requirements.txt --upgrade

if [[ ! -d "dask_space" ]]; then
    mkdir dask_space
fi

cd dask_space || exit 1

echo "loading dask from:"
which dask-scheduler

dask-scheduler &

dask-worker --no-nanny localhost:8786

if [[ "$REMOVE_AFTER_ALL" -eq "1" ]]; then
  cd ..
  rm -fr ./venv
  rm -fr ./dask
  rm -fr ./distributed
fi