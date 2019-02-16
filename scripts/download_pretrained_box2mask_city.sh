if [ ! -d checkpoints/pretrained_box2mask_city/ ]; then
  mkdir checkpoints/pretrained_box2mask_city/
fi

wget https://umich.box.com/shared/static/4ykks8aou78smmpn5235f0bzml8ehixg.pth -cO checkpoints/pretrained_box2mask_city/latest_net_G.pth

