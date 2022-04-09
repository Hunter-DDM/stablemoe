# stablemoe
Code for the ACL-2022 paper "StableMoE: Stable Routing Strategy for Mixture of Experts"

## Install Customized Fairseq
This project is developed based on the [fairseq](https://github.com/pytorch/fairseq) codebase. Before running the code, please first run the following command in the root directory of this project to install our customized fairseq:
```
pip install --user --editable .
python setup.py build_ext --inplace
```

## Prepare Data

For language model, you can preprocess your own corpus with the ```fairseq-preprocess``` command. Please refer to the [official example](https://github.com/pytorch/fairseq/tree/main/examples/language_model) in fairseq for more details. If the corpus is too large, you may need to split it into shards for better efficiency. 

## Train a StableMoE Language Model

An example training command to train a 16-expert StableMoE model (with 16 GPUs) is as follows: 
```
DATADIR=(the directory that saves the preprocessed data)
jobname=(the job name)
mkdir -p ../checkpoints/$jobname
python -m torch.distributed.launch \
    --nproc_per_node=16 \
    train.py $DATADIR \
    --task language_modeling \
    --save-dir ../checkpoints/$jobname \
    --arch transformer_lm_BaseGPT_x1_small \
    --moe-type base_layer \
    --two-stage-updates 6000 \
    --distill-assignment \
    --distilled-model wordemb \
    --distill-factor 0.3 \
    --criterion xentropy_aux \
    --balance-loss balance \
    --balance-factor 0.3 \
    --capacity-factor 2 \
    --assignment-algorithm GA \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 \
    --lr 0.0006 \
    --lr-scheduler polynomial_decay \
    --total-num-update 60000 \
    --warmup-updates 2000 \
    --tokens-per-sample 1024 \
    --sample-break-mode none \
    --batch-size 8 \
    --pad-to-fixed-length \
    --pad-to-fixed-bsz \
    --update-freq 4 \
    --max-update 60000 \
    --ddp-backend=legacy_ddp \
    --log-interval 100 \
    --log-file ../checkpoints/$jobname/log.txt \
    --log-format tqdm \
    --validate-interval-updates 500 \
    --save-interval 5 \
    --tensorboard-logdir ../tblogs/$jobname \
    --distributed-no-spawn \
    --fp16-no-flatten-grads \
    --fp16
```

## Evaluate the Model
After training, an example command to evaluate a 16-expert StableMoE model is as follows: 
```
python -m torch.distributed.launch \
    --nproc_per_node=16 \
    fairseq_cli/eval_lm.py $DATADIR \
    --path ../checkpoints/$jobname/checkpoint_best.pt \
    --tokens-per-sample 1024 \
    --batch-size 8 \
    --ddp-backend=legacy_ddp \
    --distributed-no-spawn \
    --fp16-no-flatten-grads \
    --fp16 \
    --save-each-rank
```

## Citation

If you use this code for your research, please kindly cite our ACL-2022 paper:
```
will be specified later
```

## Contact

Damai Dai: daidamai@pku.edu.cn
Li Dong: lidong1@microsoft.com
