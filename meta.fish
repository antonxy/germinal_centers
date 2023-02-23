#!/usr/bin/env fish

cd /opt/daria_microscopy/Emile

for f in **
	echo $f
	if test -f $f
		mkdir -p ../meta/(dirname $f)
		~/daria/microscopy/py/check_metadata.py $f -c ../meta/$f.csv -o ../meta/$f.xml
	end
end
