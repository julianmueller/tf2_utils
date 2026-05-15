# TODO

- add utility functions for:
    - ~~aligning a tf's (x/y/z) axis towards a point/tf~~
    - ~~aligning a tf's (xy/yz/xz) plane with:~~
        - a point and a normal
        - parallel to another tfs plane

    - ~~align parallel a tf's axis to another tf's axis~~
    - ~~align in dir of anothers tfs axis with an offset (0 offset means pointing at the other tf, Point(x=1.2) is 1.2 into the dir of that tf's x-axis)~~
        
    - ~~reparent a tf to a different parent, but keping the "global tf" the same. this necessitates to compute the chain for this to~~

    - add a convenience method to quickly add new frames in a tree.
    tree = (name="", tf=Transform(), children=[(name="", tf=Transform(), children=[]),()])