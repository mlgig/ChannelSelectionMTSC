datadir=./data
tempres=./temp 
fn=eu
str=all
#choice=True False
#center=median mad mean median

#echo MrSEQL_${fn}_${str}

python main/dtw.py --datadir ./data/ --tempres $tempres --strategy all>logs_dtw/dtw_all
python ../Generic_Scripts/combine_csv.py $tempres dtw_all

for mc in True False
do
	for fft in True False
	do
		for strategy in ecs ecp
		do
			for c in mean median mad
			do
				python -W ignore -u main/dtw.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $strategy --center $c --mc $mc --fft $fft > logs_dtw/dtw_${fn}_${strategy}_${c}_${mc}_${fft}
				python ../Generic_Scripts/combine_csv.py $tempres dtw_${fn}_${strategy}_${c}_${mc}_${fft}
				#rm $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*
				
			done
		done

	done
done
