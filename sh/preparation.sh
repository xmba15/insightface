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

wget https://www.dropbox.com/s/2r5tg6e7n95x9q3/model-r50-am-lfw.zip -P $models_path
unzip $models_path/model-r50-am-lfw.zip -d $models_path
wget https://www.dropbox.com/s/ytpomoqjed25p1n/model-r100-arcface-ms1m-refine-v2.zip
unzip $models_path/model-r100-arcface-ms1m-refine-v2.zip -d $models_path
