#!/bin/bash

# Default values for the arguments that don't change across iterations
num_examples=100
model="gemma-2-2b"
dict_path="dictionaries" #TODO understand if right
dict_id=10
dict_size=32768
batch_size=10
aggregation="none"
method="attrib"
node_threshold=0.1
max_features=32768
pen_thickness=10
nopair=false
plot_circuit=true
nodes_only=true
plot_only=false
circuit_dir="circuits"
plot_dir="circuits/figures"
seed=12
device="cuda:0"
dataset="ioi_examples"
example_length=15

# Run the Python script with arguments
python circuit.py \
    --dataset $dataset \
    --num_examples $num_examples \
    --example_length $example_length \
    --model $model \
    --dict_path $dict_path \
    --dict_id $dict_id \
    --method $method \
    --dict_size $dict_size \
    --batch_size $batch_size \
    --method $method \
    --aggregation $aggregation \
    --node_threshold $node_threshold \
    --max_features $max_features \
    --pen_thickness $pen_thickness \
    $( [ "$nopair" = true ] && echo "--nopair" ) \
    $( [ "$plot_circuit" = true ] && echo "--plot_circuit" ) \
    $( [ "$nodes_only" = true ] && echo "--nodes_only" ) \
    $( [ "$plot_only" = true ] && echo "--plot_only" ) \
    --circuit_dir $circuit_dir \
    --plot_dir $plot_dir \
    --seed $seed \
    --device $device

done