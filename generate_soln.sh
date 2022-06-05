#!/bin/bash

# Simple bash script to run the solution agent on every level

for i in {0..255}
do
    python3 "agents/solution_agent.py" "$i"
done