conda create -n mstar python=3.8
conda activate mstar
pip install -r requirements.txt

#for training
curate training datasets datasets/syn100k_mlt_word.json
