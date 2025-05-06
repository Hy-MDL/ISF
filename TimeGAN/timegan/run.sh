#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="test"

# Iteration variables
emb_epochs=30
sup_epochs=30
gan_epochs=30

python /home/hyeonmin/ISF/timegan-pytorch-main/main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       24 \
--train_rate        0.01 \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        64 \
--hidden_dim        32 \
--num_layers        4 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     5e-4 \