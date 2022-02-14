#!/bin/bash

datadir=./data/
tempres=tempSL 
fn=eu
str=all
note=mad
#center=median mad mean median

#echo MrSEQL_${fn}_${str}

function script()
{
    local dir=tempSL_$1_$2
    if [ ! -d $dir ];
    then
        mkdir $dir
        echo file created
    fi
    #echo $dir
    echo $1_$2 running
    python -W ignore -u main/MrSEQL.py --datadir $datadir --tempres $dir --strategy $2 --center $1  > logs_seql/seql_$1_$2_$note
    python ../Generic_Scripts/combine_csv.py $dir SEQL_$2_$1_$note
    #echo  $datadir  $tempres $fn $strategy $c
    #echo WM2_${fn}_$2_$1
    echo $1_$2 completed
    
}

for c in mad #mean median
do
    for strategy  in ecp ecs 
    do
        script "$c" "$strategy" &
    done
done

exit 0



#for c in mean median mad
#do
#    for strategy  in ecp km ecs 
#    do
#        python -W ignore -u main/WM.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $strategy --center $c # > logs_wm/WM_${strategy}_${c}
#        #python ../Generic_Scripts/combine_csv.py $tempres WM_${fn}_${strategy}_${c}
#    	#echo Rocket_${fn}_${strategy}_${c}
#    	#rm $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*
#    done
#done

