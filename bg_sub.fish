#!/usr/bin/env fish

cd /opt/daria_microscopy/Emile

for f in **
	echo $f
	if test -f $f
		mkdir -p ../bg_sub/(dirname $f)
		~/daria/microscopy/py/mosaic_background_subtract.py -d 16 $f ../bg_sub/$f
	end
end
