# Seq2Seq: Cold and Dynamic Fusion

Project Implementation of Cold Fusion (Sriram et al., 2017: Cold Fusion: Training Seq2Seq Models Together with Language Models) 
and Dynamic Fusion (Kurosawa and Komachi, 2019: Dynamic Fusion: Attentional Language Model for Neural Machine Translation)
into OpenNMT-py for the seminar 'Recent advances in sequence-to-sequence learning'.

All trained models, the outputs and training logs can be found on Cluster under /home/students/meier/Seq2Seq/final_models

In order to run the code, Python 3.5.3 or higher is required as well a installation of Pytorch and torchtext 0.4.0.

# Results #

Results with Gigaword LM

| Model  | BLEU score |
| ------------- | ------------- |
| Baseline  | 26.77  |
| Baseline + Cold Fusion | 16.76 |
| Baseline + Cold Fusion Variant 1| 18.02 |
| Baseline + Dynamic Fusion | 23.32 |


Results with Europarl LM

| Model  | BLEU score |
| ------------- | ------------- |
| Baseline  | 24.64  |
| Baseline + Cold Fusion | 22.05 |
| Baseline + Cold Fusion Variant 1| 21.16 |
| Baseline + Dynamic Fusion | 23.45 |

# Training Times #

Training times of models using the Gigaword language model

| Model  | Number of Epochs |Training Time |
| ------------- | ------------- | ------------- |
| Baseline  | 100,000 | 05:29:52 |
| Baseline + Cold Fusion | 200,000 | 1-03:52:21 |
| Baseline + Cold Fusion Variant 1| 200,000 | 1-02:52:09|
| Baseline + Dynamic Fusion | 200,000 | 22:00:39 |


Training times of models using the Europarl language model

| Model  | Number of Epochs |Training Time |
| ------------- | ------------- | ------------- |
| Baseline  | 200,000 | 11:05:45 |
| Baseline + Cold Fusion | 200,000 | 1-06:27:15 |
| Baseline + Cold Fusion Variant 1| 200,000 | 1-06:21:35|
| Baseline + Dynamic Fusion | 200,000 | 24:21:31 |

# Reproduce the Results #

### Baseline ###


Train the model
```
cd OpenNMT
python3 train -config config/baseline.yml -data data/final/europarl 
              -save_model baseline-final 
```

Run inference on test set
```
python3 translate.py -beam_size 5 -model baseline_europarl_step_200000.pt -src data/final/test.de 
                        -output data/bl_pred.txt -replace_unk -verbose
```

Get final bleu score

```
srun sed -i "s/@@ //g" data/bl_pred.txt
srun perl tools/multi-bleu.perl data/final/test.en < data/bl_pred.txt
```


### Cold Fusion ###

Train the model

```
cd OpenNMT
python3 train -config config/config_cold_fusion.yml -data data/final/europarl
               -save_model cold_fusion  -language_model ../language_model/gru_best_perplexity_modelGRU.pth
```

Run inference on test set
```
python3 translate.py -beam_size 5 -model cold_fusion_step_200000.pt -src data/final/test.de  -output data/test_cf.txt -replace_unk -verbose
```

Get final bleu score
```
srun sed -i "s/@@ //g" data/test_cf.txt > data/predictions.txt
srun perl tools/multi-bleu.perl data/final/test.en < data/test_cf.txt
```


### Cold Fusion Variant 1 ###



applies $`s_t`$ to gating g instead of $`h_t^{LM}`$ (4c)

Train the model
```
cd OpenNMT
python3 train -config config/config_cold_fusion_variant1.yml -data data/final/europarl 
              -save_model cold_fusion_var1  -language_model ../language_model/gru_best_perplexity_modelGRU.pth
```
Run inference on test set
```
python3 translate.py -beam_size 5 -model cold_fusion_var1_step_200000.pt -src data/final/test.de  -output data/test_var1.txt -replace_unk -verbose
```

Get final bleu score
```
srun sed -i "s/@@ //g" data/test_var1.txt 
srun perl tools/multi-bleu.perl data/final/test.en < data/test_var1.txt
```

### Dynamic Fusion ###


Train the model
```
cd OpenNMT
srun python3 train.py -config config/config_dynamic_fusion.yml -data /data/final/europarl 
                      -save_model dynamic_europarl -language_model ../language_model/gru_best_perplexity_modelGRU.pth
```

Run inference on test set
```
srun python3 translate.py -beam_size 5 -model dynamic_europarl_step_200000.pt -src /data/final/test.de -output data/eval_dyn_final.txt -replace_unk -verbose
```

Get final bleu score
```
srun sed -i "s/@@ //g" data/eval_dyn_final.txt 
srun perl tools/multi-bleu.perl data/final/test.en < data/eval_dyn_final.txt
```


# Data #

Parallel data: Europarl Corpus (Koehn, 2005:Europarl: A Parallel Corpus for Statistical Machine Translation
)

Monolingual data: 2 Million sentences of the Gigaword  corpus(Graff et al., 2003: )

# Preprocessing #

The data was cleaned before processing. Unprintable tokens, index digits and brackets were removed and everything was set to lowercase. Also a length constraint was introduced, 
sentences with more than 60 words were removed and sentence pairs which had length difference of more than 15 words were removed.



