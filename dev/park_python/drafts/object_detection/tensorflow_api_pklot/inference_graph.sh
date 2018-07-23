python /home/ubuntu/tensorflow_models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=training.config \
--trained_checkpoint_prefix=/home/ubuntu/DS/PKLot/tensorflow_ds/training_pklotfull_4000/model.ckpt-4000 \
--output_directory=/home/ubuntu/DS/PKLot/tensorflow_ds/inference_pklot_full_4000

