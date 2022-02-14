datadir=./data/
tempres=./tempSEQL
fn=eu
str=all

python -W ignore -u main/MrSEQL.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $str > logs_seql/SEQL_${str}
python ../Generic_Scripts/combine_csv.py $tempres SEQL_${str}


