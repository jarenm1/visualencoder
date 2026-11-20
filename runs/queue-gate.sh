#!/usr/bin/env bash
# Validate the encoder gate on two known cases.
cd /home/jaren/visualencoder
G="nix develop --command nix develop path:sim --command env PYTHONPATH=sim/python:src .venv/bin/python -u sim/python/encoder_gate.py"
$G --baseline run:runs/mean --candidate run:runs/posttrain-mean-b4 --out reports/gate/posttrain-mean-b4 > reports/gate-a.log 2>&1
echo "A exit $?" >> reports/gate-a.log
$G --baseline lingbot:small --candidate run:runs/mean --out reports/gate/mean-vs-small > reports/gate-b.log 2>&1
echo "B exit $?" >> reports/gate-b.log
