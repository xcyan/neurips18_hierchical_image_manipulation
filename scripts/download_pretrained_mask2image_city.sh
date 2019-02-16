if [ ! -d checkpoints/pretrained_mask2image_city/ ]; then
  mkdir checkpoints/pretrained_mask2image_city/
fi

wget https://umich.box.com/shared/static/jd9p3urdss97bnqxi2wd25ru6cxvsvv7.pth -cO checkpoints/pretrained_mask2image_city/latest_net_G.pth

