#! /bin/bash
gpus=$1
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests            GL3D --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests           KITTI --img_size 512 
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests          GTASfM --img_size 512 
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests         ICLNUIM --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests        MultiFoV --img_size 512 