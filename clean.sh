set -x

rm -rf train/*
rm -rf tensorboard/*

cache_files=`find -name __pycache__`
for file in ${cache_files}
do
    rm -rf ${file}
done
