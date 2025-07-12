#!/bin/bash
cd /dynamic_iqa_ex/ddfnet/ddf
python setup.py install
mv build/lib*/* .
cd /dynamic_iqa_ex/
exec "$@"
