#!/bin/bash
# jupyter notebook password
# jupyter notebook list
: "${JPORT:=8870}"
module add openssl-1.1.1
module add readline-8.0
module add zlib-1.2.11
module add libffi-3.2.1
module add sqlite-3.29.0
module add graphviz-2.26.3
module add bzip2-1.0.8
module add lzma-4.32.7
module add mariadb-client-8.0.17    # module created from https://dev.mysql.com/downloads/mysql/    Linux - Generic (glibc 2.12) (x86, 64-bit), Compressed TAR Archive
jupyter-notebook  --no-browser --port $JPORT .
