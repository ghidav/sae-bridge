#!/bin/bash

# Default values for the arguments that don't change across iterations
num_examples=100
model="pythia-70m-deduped"
dict_path="dictionaries" #TODO understand if right
dict_id=10
dict_size=32768
batch_size=10
aggregation="none"
node_threshold=0.1
max_features=32768
pen_thickness=10
nopair=false
plot_circuit=true
nodes_only=true
component_level=true
plot_only=false
circuit_dir="circuits"
plot_dir="circuits/figures"
seed=12
device="cuda:0"
method="exact"
dataset="rc_train"
example_length=6

# Run the Python script with arguments
python circuit.py \
    --dataset $dataset \
    --num_examples $num_examples \
    --example_length $example_length \
    --model $model \
    --dict_path $dict_path \
    --dict_id $dict_id \
    --dict_size $dict_size \
    --batch_size $batch_size \
    --aggregation $aggregation \
    --node_threshold $node_threshold \
    --method $method \
    --max_features $max_features \
    --pen_thickness $pen_thickness \
    $( [ "$component_level" = true ] && echo "--component_level" ) \
    $( [ "$nopair" = true ] && echo "--nopair" ) \
    $( [ "$plot_circuit" = true ] && echo "--plot_circuit" ) \
    $( [ "$nodes_only" = true ] && echo "--nodes_only" ) \
    $( [ "$plot_only" = true ] && echo "--plot_only" ) \
    --circuit_dir $circuit_dir \
    --plot_dir $plot_dir \
    --seed $seed \
    --device $device

done