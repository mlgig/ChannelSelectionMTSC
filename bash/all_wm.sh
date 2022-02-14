datadir=./data/
tempres=./tempWM
fn=eu
str=all

python -W ignore -u main/WM.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $str > logs_wm/WM_${str}
python ../Generic_Scripts/combine_csv.py $tempres WM_${str}

