#!/bin/bash

datadir=./data/
tempres=tempWM 
fn=eu
str=all
file=WM
note=median #_l2

#center=median mad mean median

#echo MrSEQL_${fn}_${str}

function script()
{
    #echo ${tempres}_$1_$2..............
    #echo logs_${file}
    local dir=${tempres}_$1_$2
    if [ ! -d $dir ];
    then
        mkdir $dir
        echo $dir file created
    fi
    local dir2=logs_${file} 
    if [ ! -d $dir2 ];
    then
        mkdir $dir2
        echo $dir2 file created
    fi
    #echo $dir
    echo ${file}_$1_$2 running
    #echo ${file}_${fn}_$2_$1$note
    python -W ignore -u main/${file}.py --datadir $datadir --tempres $dir --strategy $2 --center $1 --mc True  > logs_${file}/${file}_${fn}_$2_$1$note 
    python ../Generic_Scripts/combine_csv.py $dir ${file}_${fn}_$2_$1$note
    #echo  $datadir  $tempres $fn $strategy $c
    #echo WM2_${fn}_$2_$1
    echo ${file}_$1_$2 completed

    rm -rf $dir
    
}

for c in median #mean mad
do
    for strategy  in ecp ecs 
    do
        script "$c" "$strategy" &
    done
done

exit 0
