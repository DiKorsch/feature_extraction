#!/usr/bin/env bash

./extract_GT_grouped_inceptionV3_inat_TF.sh

./extract_GT_inceptionV3_inat_TF.sh

FULL=0 ./extract_L1_inceptionV3_inat_TF.sh

FULL=1 ./extract_L1_inceptionV3_inat_TF.sh

./extract_NAC_inceptionV3_inat_TF.sh
