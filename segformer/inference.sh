python inference.py \
  --weights logs/segformer_b0_wide/best \
  --input /home/seungbo/RemovalAI/HierarchicalDataset/test/RGB \
  --output-dir outputs/predictions_wide_whole_data \
  --device cuda \
  --save-video