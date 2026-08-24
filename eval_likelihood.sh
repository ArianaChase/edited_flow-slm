output_dir="$root_dir/work/outputs/flow-slm"

if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi
root_dir="/home/ubuntu/speech_ppl"
conf_path="$root_dir/flow-slm/conf/270m.yaml"
data_dir="/share/data/speech/jjery2243542/data/salmon"
id="/share/data/speech/jjery2243542/data/continuous_gslm/salmon/test.txt" # the id file for evaluation, each line is a relative path from data_dir, with the extension defined in conf.yaml
ckpt_path="$root_dir/work/pretrained_models/flow-slm/270m.bin"
k_future_tokens=4
batch_size=1

python $root_dir/flow-slm/trainer.py \
    --data_dir $data_dir \
    --conf $conf_path  \
    --predict_id_file $id \
    --override "{'model': {'flash_attention': False}, 'training':{'batch_size': $batch_size}, 'data': {'ext': 'wav'}}" \
    --ckpt_path $ckpt_path \
    --is_speechocean \
    --reduction "token_seq" \
    --predict_only \
    --prediction_output_dir $output_dir \
    --use_k_future_tokens $k_future_tokens \
    --ignore_eos \
    --root_dir $root_dir \
