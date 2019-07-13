#!/bin/bash

os_system=`uname`
if [ $os_system == "Darwin" ]; then
    realpath() {
        path=`eval echo "$1"`
        folder=$(dirname "$path")
        echo $(cd "$folder"; pwd)/$(basename "$path");
    }
fi

absolute_path=`pwd`/`dirname $0`
models_path=`realpath $absolute_path/../models`

wget https://www.dropbox.com/s/35yxhevcqijddu4/model-r50-am-lfw.zip -P $models_path
unzip $models_path/model-r50-am-lfw.zip -d $models_path
wget https://www.dropbox.com/s/1z274ao8lz1c0m2/model-r100-arcface-ms1m-refine-v2.zip -P $models_path
unzip $models_path/model-r100-arcface-ms1m-refine-v2.zip -d $models_path
