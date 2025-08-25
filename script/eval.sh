exp_name=$1

python eval.py -d ./result/${exp_name}
for constraint in 'elements' 'formula'
do
    python eval.py -d ./result/${exp_name} -c ${constraint}
done

# sh script/eval.sh demo