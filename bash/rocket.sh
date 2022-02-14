datadir=./data
tempres=./temp 
fn=eu
str=all
#choice=True False
#center=median mad mean median

#echo MrSEQL_${fn}_${str}

for mc in True False
do
	for fft in True False
	do
		for strategy in ecs ecp
		do
			for c in mean median mad
			do
				python -W ignore -u main/Rocket.py --datadir $datadir --tempres $tempres --distancefn $fn --strategy $strategy --center $c --mc $mc --fft $fft > logs_mag/Rocket_mag_${fn}_${strategy}_${c}_${mc}_${fft}
				python ../Generic_Scripts/combine_csv.py $tempres Rocket_${fn}_${strategy}_${c}_${mc}_${fft}
				rm $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*
				
			done
		done



	done
done


