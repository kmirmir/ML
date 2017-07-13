for (( layer=1; layer<=7; layer++ ))
do
    for (( learning=1; learning<=100; learning++))
    do
        python3.6 executeRNN.py "$layer" "$learning"
    done
done

echo " all process is done "