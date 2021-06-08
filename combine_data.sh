#!/bin/bash

i=0
k=1
for arg in $*
do
    if [ $i == 0 ]
    then
        target_dir=$arg
    else
        src_dir=$arg
        echo $src_dir
        cp $src_dir/params.txt $target_dir
        j=1
        while true
        do
            if test -f "$src_dir/label$j.png"
            then
                cp $src_dir/$j.png $target_dir/$k.png
                cp $src_dir/label$j.png $target_dir/label$k.png
                ((j++))
                ((k++))
            else
                break
            fi
        done
    fi
    ((i++))
done
