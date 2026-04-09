#!/bin/bash

i=1
for f in *; do
  ext="${f##*.}"
  mv -- "$f" "name-$i.$ext"
  ((i++))
done