#!/usr/bin/env bash

pathToDataOnNAF=/nfs/dust/cms/user/matthies/uhh2-106X-v2/CMSSW_10_6_28/src/UHH2/HighPtSingleTop/output/Analysis/mainsel/run2/both/nominal/hadded

dirRawData=data/raw
echo "Create directory: ${dirRawData}"
mkdir -p ${dirRawData}
echo "Copy files"
scp naf-cms:${pathToDataOnNAF}/uhh2.AnalysisModuleRunner.MC.*.root ${dirRawData}/.
echo "Done"
