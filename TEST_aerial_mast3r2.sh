#! /bin/bash
gpus=$1 
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests      BlendedMVS --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests        SceneNet --img_size 512 
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests          ETH3DI --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests          ETH3DO --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests   RobotcarNight --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests  RobotcarSeason --img_size 512
python test.py --gpus $gpus --weight mast3r_onlymega --version 100h --test --batch_size 1 --tests RobotcarWeather --img_size 512