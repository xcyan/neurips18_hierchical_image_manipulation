if [ ! -d checkpoints/pretrained_box2mask_ade/ ]; then
  mkdir checkpoints/pretrained_box2mask_ade/
fi

wget https://umich.box.com/shared/static/j65mss4ygl5iokrare5r5joefggnciwg.pth -cO checkpoints/pretrained_box2mask_ade/latest_net_G.pth

