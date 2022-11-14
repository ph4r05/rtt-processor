#!/bin/bash
: "${JPORT:=8870}"
ssh -L $JPORT:localhost:$JPORT -A aura -t 'jupyter notebook list; bash -l'
