names=`ls ./processed`
for name in $names
do
    echo $name
    python 2_project.py --n $name
done