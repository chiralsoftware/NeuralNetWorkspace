#!/bin/bash

echo "Downloading the MINST data"
echo "Based on URLs in:"
echo "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py"

BASE_URL=https://storage.googleapis.com/cvdf-datasets/mnist/
declare -A FILES
FILES=(
  ["train-images-idx3-ubyte.gz"]="f68b3c2dcbeaaa9fbdd348bbdeb94873"
  ["train-labels-idx1-ubyte.gz"]="d53e105ee54ea40749a09fcbcd1e9432"
  ["t10k-images-idx3-ubyte.gz"]="9fb629c4189551a2d022fa330f9573f3"
  ["t10k-labels-idx1-ubyte.gz"]="ec29112dd5afa0611ce80d1b7f02629c"
)

for FILE in "${!FILES[@]}"; do
  if [[ -f "$FILE" ]]; then
    echo "Checking MD5 for $FILE..."
    MD5_LOCAL=$(md5sum "$FILE" | awk '{print $1}')
    MD5_EXPECTED="${FILES[$FILE]}"
    if [[ "$MD5_LOCAL" == "$MD5_EXPECTED" ]]; then
      echo "$FILE is already downloaded and verified."
      continue
    else
	echo "$FILE exists but failed MD5 check. Re-downloading..."
	rm -f $FILE
    fi
  else
    echo "$FILE not found. Downloading..."
  fi
  wget "${BASE_URL}${FILE}" -O "$FILE"
done
