set -x

base_file="./docs/stock_selection_base.md"
tmp_file="./docs/stock_selection.md.tmp"
target_file="./docs/stock_selection.md"
probidx=5

# initialize target file
cp ${base_file} ${target_file}
python test.py --docs ${tmp_file} --thresh 0.8

cat ${tmp_file} | sort -r -t "|" -k ${probidx} >> ${target_file} && rm -f ${tmp_file}
cd ./data/ && python statistics.py && cd -

echo "Finished processing ${target_file}!!"
#echo `head -n 40 ${target_file}`
