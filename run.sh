#!/bin/bash
# DeiT CIFAR-100: Experiments
# Runs 4 variations: {Pretrained, No Pretrained} x {Baseline, Pruned}

set -e
export PYTHONPATH=.
mkdir -p logs

echo "======================================================================="
echo "  DeiT CIFAR-100: Pruning vs Baseline Experiments"
echo "======================================================================="

# Function to run experiment
run_exp() {
    PRE=$1
    COMP=$2
    TAG="pretrained"
    if [ "$PRE" == "no" ]; then TAG="no_pretrained"; fi

    TYPE="baseline"
    if [ "$COMP" == "yes" ]; then TYPE="pruned"; fi

    echo ""
    echo "▶ Running: Pretrained=$PRE | Compress=$COMP"
    python awnn_cifar100.py --pretrained $PRE --compress $COMP 2>&1 | tee logs/${TAG}_${TYPE}.log
}

# 1. No Pretrain | Baseline
run_exp no no

# 2. No Pretrain | Pruned
run_exp no yes

# 3. Pretrained | Baseline
run_exp yes no

# 4. Pretrained | Pruned
run_exp yes yes

echo ""
echo "======================================================================="
echo "✅ All Experiments Complete!"
echo "======================================================================="

# Generate report
echo ""
echo "SUMMARY REPORT:"
echo "---------------"

grep "Final Val Accuracy:" logs/*.log | while read line; do
    FILE=$(echo $line | cut -d: -f1)
    ACC=$(echo $line | cut -d: -f3)

    # Extract Pruned % (only for pruned runs)
    PRUNED="0.0%"
    if [[ $FILE == *"_pruned.log" ]]; then
        PRUNED=$(grep "Compression:" $FILE | tail -n 1 | awk '{print $NF}')
    fi

    echo "$FILE:"
    echo "  Accuracy: $ACC"
    echo "  Pruned:   $PRUNED"
    echo ""
done
