#!/bin/bash

N=10

for nEdg in 1 2 
do
	for cen in "c_pr" "c_nxkatz"
	do
		for isDir in 1 0
		do
			python3 centralityProfileRun.py -o data100e_N${N}_${cen}_nEdg${nEdg}_Dir${isDir}.pkl -N ${N} -Dir ${isDir} -c ${cen} 
		done
	done
done
