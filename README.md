This is the code and data of our NAACL 2024 paper: zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models.

#### Dataset
Datasets are in the data.zip file. Each dataset is under the directory data/${DATASET}/.
The enriched relation descriptions are also under data/${DATASET}/. If you wish to generate LLM-based relation representations yourself, go to data/ and run:

`python generate_emb.py`

#### Software
The code of seven baseline TKG forecasting methods equipped with zrLLM are in the software.zip file. Due to the size limit of code and data on openreview, we only provide the generated LLM-based relation representations of ACLED. If you wish, you can generate representations of ICEWS21 and ICEWS22 by yourself. 

#### How To Run
**CyGNet**
1. Install environment with all required dependencies specified at https://github.com/CunchaoZ/CyGNet.
2. Copy all the files under data/${DATASET}/ to software/CyGNet/${DATASET}/.
3. Get historical vocabulary as instructed in CyGNet repository.
   
   `python get_historical_vocabulary.py --dataset ${DATASET}`
4. Go to software/CyGNet/. For each dataset, do zrLLM preprocessing by running:
   
   `python get_history.py --dataset ${DATASET}`
5. To experiment zrLLM-enhanced CyGNet, for each dataset, go to software/CyGNet/ and run:
   
   `python train.py --dataset ACLED --entity object --time-stamp 1 --alpha 0.8 --lr 0.001 --n-epoch 50 --hidden_dim 200 --embedding_dim 200 --gpu 0 --batch-size 256 --counts 10 --valid-epoch 0 --withRegul --LLM_init --pure_LLM --LLM_path --path --gamma_init 1`
   
   `python train.py --dataset ICEWS21 --entity object --time-stamp 1 --alpha 0.8 --lr 0.001 --n-epoch 50 --hidden_dim 200 --embedding_dim 200 --gpu 0 --batch-size 256 --counts 10 --valid-epoch 0 --withRegul --LLM_init --pure_LLM --LLM_path --path --gamma_init 0.001`
   
   `python train.py --dataset ICEWS22 --entity object --time-stamp 1 --alpha 0.8 --lr 0.001 --n-epoch 50 --hidden_dim 200 --embedding_dim 200 --gpu 0 --batch-size 256 --counts 10 --valid-epoch 0 --withRegul --LLM_init --pure_LLM --LLM_path --path --gamma_init 0.001`

**TANGO-TuckER/Distmult**
1. Install environment with all required dependencies specified at https://github.com/TemporalKGTeam/TANGO (remember to change the torchdiffeq as instructed in the repository).
2. Copy all the files under data/${DATASET}/ to software/TANGO/${DATASET}/.
3. Do preprocessing for each dataset as instructed in TANGO repository.
4. Go to software/TANGO/. For each dataset, do zrLLM preprocessing by running:
   
   `python get_history.py --dataset ${DATASET}`
5. To experiment zrLLM-enhanced TANGO, for each dataset, go to software/TANGO/ and run:

   `python TANGO.py --dataset ${DATASET} --adjoint_flag --res --embsize ${EMBSIZE} --initsize ${EMBSIZE} --hidsize ${EMBSIZE} --score_func tucker --device ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init ${GAMMA} --gamma_fix ${FIX}`

   `python TANGO.py --dataset ${DATASET} --adjoint_flag --res --embsize ${EMBSIZE} --initsize ${EMBSIZE} --hidsize ${EMBSIZE} --score_func distmult --device ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init ${GAMMA} --gamma_fix ${FIX}`
   
   Please use the hyperparameters reported in Appendix C. Set --gamma_fix to 1 if you want to fix gamma, otherwise set to 0.

**RE-GCN**
1. Install environment with all required dependencies specified at https://github.com/Lee-zix/RE-GCN.
2. Copy all the files under data/${DATASET}/ to software/RE-GCN/data/${DATASET}/.
3. Go to software/RE-GCN/src/. For each dataset, do zrLLM preprocessing by running:
	
   `python get_history.py --dataset ${DATASET}`
4. For ICEWS22-zero, do static graph preprocessing. Go to software/RE-GCN/data/ICEWS22/ and run:
   
   `python ent2word.py`
5. To experiment zrLLM-enhanced RE-GCN, go to software/RE-GCN/src/ and run:
   
   `python main.py -d ACLED --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --task-weight 1 --gpu ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init 0.01`
   
   `python main.py -d ICEWS21 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --task-weight 1 --gpu ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init 0.01`

   `python main.py -d ICEWS22 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --task-weight 1 --gpu ${GPU} --add-static-graph --angle 10 --discount 1 --LLM --pure_LLM --path --LLM_path --gamma_init 0.01`
   
**TiRGN**
1. Install environment with all required dependencies specified at https://github.com/Liyyy2122/TiRGN.
2. Copy all the files under data/${DATASET}/ to software/TiRGN/data/${DATASET}/.
3. Go to software/TiRGN/src/. For each dataset, do zrLLM preprocessing by running:
   
   `python get_history.py --dataset ${DATASET}`
4. For ICEWS22-zero, do static graph preprocessing. Go to software/TiRGN/data/ICEWS22/ and run:

   `python ent2word.py`
5. To experiment zrLLM-enhanced TiRGN, go to software/TiRGN/src/ and run:
   
   `python main.py -d ACLED_final --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --entity-prediction --task-weight 1 --gpu ${GPU} --save checkpoint --LLM --pure_LLM --path --LLM_path --gamma_init 0.001 --gamma_fix 0`
   
   `python main.py -d ICEWS21 --history-rate 0.3 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 100 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --entity-prediction --task-weight 1 --gpu ${GPU} --save checkpoint --LLM --pure_LLM --path --LLM_path --gamma_init 0.01 --gamma_fix 1`
   
   `python main.py -d ICEWS22 --history-rate 0.3 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 --entity-prediction --add-static-graph --angle 14 --discount 1 --task-weight 1 --gpu ${GPU} --save checkpoint --LLM --pure_LLM --path --LLM_path --gamma_init 0.01 --gamma_fix 1`
   
**RETIA**
1. Install environment with all required dependencies specified at https://github.com/CGCL-codes/RETIA.
2. Copy all the files under data/${DATASET}/ to software/RETIA/data/${DATASET}/.
3. Go to software/RETIA/src/. For each dataset, do zrLLM preprocessing by running:
   
   `python get_history.py --dataset ${DATASET}`
4. For ICEWS22-zero, do static graph preprocessing. Go to software/RETIA/data/ICEWS22/ and run:
   
   `python ent2word.py`
5. To experiment zrLLM-enhanced RETIA, go to software/RETIA/src/ and run:
   
   `python main.py -d ACLED --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu ${GPU} --ft_lr=0.001 --norm_weight 1 --task-weight 1 --LLM --pure_LLM --path --LLM_path --gamma_init 0.01 --gamma_fix 0`
   
   `python main.py -d ICEWS22 --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu ${GPU} --ft_lr=0.001 --norm_weight 1 --task-weight 1 --weight 0.5 --angle 14 --discount 1 --add-static-graph --LLM --pure_LLM --path --LLM_path --gamma_init 0.01 --gamma_fix 1`
   
**CENET**
1. Install environment with all required dependencies specified at https://github.com/xyjigsaw/CENET.
2. Copy all the files under data/${DATASET}/ to software/CENET/data/${DATASET}/.
3. Go to software/CENET/data/${DATASET}/. For each dataset, do CENET and zrLLM preprocessing by running:
   
   `python get_history_graph_1.py`

   `python get_history_graph_2.py`
   
   `python get_history.py`
4. To experiment zrLLM-enhanced CENET, go to software/CENET/ and run:
   
   `python main.py -d ACLED --description ${EXP_NAME} --max-epochs 50 --oracle-epochs 10 --valid-epochs 5 --alpha 0.2 --lambdax 2 --batch-size 512 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir ${SAVE_DIR} --eva_dir ${EVAL_DIR} --seed ${SEED} --embedding_dim 100 --gpu ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init 1`
   
   `python main.py -d ICEWS21 --description ${EXP_NAME} --max-epochs 30 --oracle-epochs 20 --valid-epochs 5 --alpha 0.2 --lambdax 2 --batch-size 512 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir ${SAVE_DIR} --eva_dir ${EVAL_DIR} --seed ${SEED} --embedding_dim 200 --gpu ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init 1`
   
   `python main.py -d ICEWS22 --description ${EXP_NAME} --max-epochs 40 --oracle-epochs 20 --valid-epochs 5 --alpha 0.2 --lambdax 2 --batch-size 512 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir ${SAVE_DIR} --eva_dir ${EVAL_DIR} --seed ${SEED} --embedding_dim 200 --gpu ${GPU} --LLM --pure_LLM --path --LLM_path --gamma_init 0.01`
