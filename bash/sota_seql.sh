datadir=./data/
tempres=./tempSEQL 
fn=eu
str=all

#echo MrSEQL_${fn}_${str}
for c in mean median mad
do
    for strategy  in ecp km ecs 
    do
        python -W ignore -u main/MrSEQL.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $strategy --center $c  > logs_seql/SEQL_${strategy}_${c}
        python ../Generic_Scripts/combine_csv.py $tempres SEQL_${fn}_${strategy}_${c}
    	#echo Rocket_${fn}_${strategy}_${c}
    	#rm $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*
    
    done
done

