#!/bin/bash

proc=${1}

nevt=${2}

random_num=$((${3} + ${4}))

ncpu=${5}

mkdir tmp_gp_mine
mv ${proc}_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz tmp_gp_mine/
cd tmp_gp_mine
tar -xaf ${proc}_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz 

./runcmsgrid.sh $nevt $random_num $ncpu

mv cmsgrid_final.lhe ../

cd ..
rm -rf tmp_gp_mine

