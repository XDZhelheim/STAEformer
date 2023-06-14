d=${1}
g=${2}

nohup python train.py -d ${d} -g ${g} > /dev/null 2>&1 &
