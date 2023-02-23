#!/usr/bin/env fish

cd /opt/daria_microscopy/bg_sub

for f in **
	echo $f
	if test -f $f
		mkdir -p ../seg/(dirname $f)
		~/daria/microscopy/py/blob_segmentation.py $f -o ../seg/$f.csv -d ../seg/(path change-extension '' $f)
	end
end
