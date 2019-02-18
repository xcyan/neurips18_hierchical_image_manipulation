python vis_box2mask.py --dataroot=datasets/cityscape/ --dataloader cityscape --name pretrained_box2mask_city \
  --prob_bg 0.1 --label_nc 35 --output_nc 35 --model AE_maskgen_twostream --which_stream obj_context \
  --batchSize 1 --first_conv_stride 1 --first_conv_size 5 --conv_size 4 --num_layers 3 --use_resnetblock 1 \
  --num_resnetblocks 1 --nThreads 2 --norm_layer batch --cond_in ctx_obj --n_blocks 6 \
  --fineSize 256 --use_output_gate --no_comb --contextMargin 2 --min_box_size 128 --max_box_size 256 \
  --phase val --how_many 200 --gpu_ids 0

