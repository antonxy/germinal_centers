Contour selection:
`python3 draw_mask.py`

Only images which have no mask yet:
`python3 draw_mask.py --only-new`

Shortcuts:
- a - add point
- d - delete point
- i - insert point
- c - clear the boundary
- m - automatically set boundary
- x - automatically set boundary with inwards offset
- `+` - add another boundary to this image
- `-` - remove the current boundary from the image
- s - save and next slide
- q - next slide without saving


Start the blob selection on all images:
`python3 manual_blobs.py`

Start the blob selection on images which have no blobs selected yet:
`python3 manual_blobs.py --only-new`

Blob selection in specific folder:
`python3 manual_blobs.py  ../images/input/Emile/CAST\ 25/**.czi`

Shortcuts:
- a - add blob
- d - delete blob
- v - view blob
- s - save and next slide
- q - next slide without saving



Completely stop program:
In the terminal press Ctrl-Z. This suspends the program. Then run `kill %1`. This actually closes the program.