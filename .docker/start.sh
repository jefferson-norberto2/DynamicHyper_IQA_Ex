#!/bin/bash
cd /workspace/ddfnet/ddf
python setup.py install
mv build/lib*/* .
exec "$@"
