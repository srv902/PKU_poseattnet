conda activate tensorflow
oarsub -t besteffort -p "gpu='YES' and host='nefgpu25.inria.fr'" -l/gpunum=3,walltime=48 experiments/bin/train_dahlia_rgb.sh