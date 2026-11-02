#!/bin/bash
# bash scripts/unlearn.sh <tofu|muse|wmdp|bios|olmo_gw> <pint|gradascent|graddiff|npo|simnpo|undial|rmu|pint_70b> [hydra overrides...]
# Env: NAME (run name), NGPU (number of GPUs), EVAL=0 (skip evaluation), EVAL_ARGS (extra overrides for src/eval.py)

set -euo pipefail

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


benchmark=${1:?benchmark}
method=${2:?method}
shift 2

task_name=${NAME:-$benchmark/$method/$(date +%Y%m%d-%H%M%S)}
output_dir=saves/unlearn/$task_name

eval_args=()
if [ -n "${EVAL_ARGS:-}" ]; then
    read -r -a eval_args <<< "$EVAL_ARGS"
fi

case $benchmark in
    tofu | bios)
        default_gpus=1
        ;;
    muse | wmdp)
        default_gpus=2
        ;;
    olmo_gw)
        default_gpus=4
        ;;
    *)
        echo "unknown benchmark: $benchmark" >&2
        exit 1
        ;;
esac
if [ "$method" = pint_70b ]; then
    default_gpus=4
    export PINT_ZERO_INIT=1
fi
num_gpus=${NGPU:-$default_gpus}


forget_split=forget01
data_split=""
shared_overrides=()
train_overrides=()
for override in "$@"; do
    case $override in
        forget_split=* | retain_split=* | holdout_split=*)
            if [ "$benchmark" = tofu ]; then
                if [ "${override%%=*}" = forget_split ]; then
                    forget_split=${override#*=}
                fi
                continue
            fi
            ;;
        data_split=*)
            data_split=${override#*=}
            shared_overrides+=("$override")
            ;;
        bios_seed=*)
            shared_overrides+=("$override")
            ;;
    esac
    train_overrides+=("$override")
done

tofu_splits=()
if [ "$benchmark" = tofu ]; then
    percent=${forget_split#forget}
    retain_split=retain$(printf %02d $((100 - 10#$percent)))
    holdout_split=holdout$percent
    tofu_splits=(
        forget_split=$forget_split
        retain_split=$retain_split
        holdout_split=$holdout_split
    )
fi


########################################################################################################################
########################################### Unlearn ####################################################################
########################################################################################################################


if [ "$num_gpus" -gt 1 ]; then
    launcher=(
        accelerate launch
        --config_file configs/accelerate/default_config.yaml
        --num_processes "$num_gpus"
        --main_process_port "$MASTER_PORT"
    )
else
    launcher=(python)
fi

"${launcher[@]}" src/train.py --config-name=unlearn.yaml \
experiment=unlearn/$benchmark/$method \
task_name=$task_name \
${tofu_splits[@]+"${tofu_splits[@]}"} \
trainer.args.eval_on_start=false \
trainer.args.eval_strategy=no \
trainer.args.do_eval=false \
paths.output_dir=$output_dir \
${train_overrides[@]+"${train_overrides[@]}"}

if [ "${EVAL:-1}" != 1 ]; then
    exit 0
fi


########################################################################################################################
########################################### Eval #######################################################################
########################################################################################################################


model_overrides=(
    model.tokenizer_args.pretrained_model_name_or_path=$output_dir
    model.model_args.attn_implementation=sdpa
)
retain_logs=()
model_paths=("$output_dir")

case $benchmark in
    tofu)
        logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_${retain_split}/TOFU_EVAL.json
        if [ -f "$logs_path" ]; then
            retain_logs=(retain_logs_path=$logs_path)
        fi
        eval_overrides=(
            experiment=eval/tofu/default
            eval=tofu_full
            "${model_overrides[@]}"
            forget_split=$forget_split
            holdout_split=$holdout_split
            ${retain_logs[@]+"${retain_logs[@]}"}
        )
        ;;
    muse)
        logs_path=saves/eval/muse_Llama-2-7b-hf_${data_split:-News}_retrain/MUSE_EVAL.json
        if [ -f "$logs_path" ]; then
            retain_logs=(retain_logs_path=$logs_path)
        fi
        eval_overrides=(
            experiment=eval/muse/default
            eval=muse_full
            "${model_overrides[@]}"
            ${shared_overrides[@]+"${shared_overrides[@]}"}
            ${retain_logs[@]+"${retain_logs[@]}"}
        )
        ;;
    wmdp)
        if [ "$method" = pint_70b ]; then
            eval_overrides=(
                experiment=eval/wmdp/default
                model=Llama-3-70B
                "${model_overrides[@]}"
                model.model_args.device_map=auto
                ${shared_overrides[@]+"${shared_overrides[@]}"}
            )
        else
            eval_overrides=(
                experiment=eval/wmdp/default
                "${model_overrides[@]}"
                ${shared_overrides[@]+"${shared_overrides[@]}"}
            )
        fi
        ;;
    bios)
        eval_overrides=(
            experiment=eval/bios/default
            bios_model=$output_dir
            ${shared_overrides[@]+"${shared_overrides[@]}"}
        )
        ;;
    olmo_gw)
        model_paths=("$output_dir"/checkpoint-* "$output_dir")
        eval_overrides=(
            experiment=eval/olmo_gw/default
            '~model.model_args.revision'
        )
        ;;
esac

if [ "$method" != pint_70b ]; then
    visible_gpus=${CUDA_VISIBLE_DEVICES:-0}
    export CUDA_VISIBLE_DEVICES=${visible_gpus%%,*}
fi

for model_path in "${model_paths[@]}"; do
    python src/eval.py --config-name=eval.yaml \
    task_name=$task_name/eval \
    model.model_args.pretrained_model_name_or_path=$model_path \
    paths.output_dir=$model_path/evals \
    "${eval_overrides[@]}" \
    ${eval_args[@]+"${eval_args[@]}"}
done
