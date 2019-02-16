if [ ! -d checkpoints/pretrained_mask2image_ade/ ]; then
  mkdir checkpoints/pretrained_mask2image_ade/
fi

wget https://umich.box.com/shared/static/mi64jtkw4n5x2wgyjsznt9cm6vjgsmbp.pth -cO checkpoints/pretrained_mask2image_ade/latest_net_G.pth

