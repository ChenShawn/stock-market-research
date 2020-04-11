set -x

rm -rf train/*
rm -rf tensorboard/*
rm -rf data/logs/*
rm nohup.out

cache_files=`find -name __pycache__`
for file in ${cache_files}
do
    rm -rf ${file}
done
