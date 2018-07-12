python /home/ubuntu/tensorflow_models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=training.config \
--trained_checkpoint_prefix=/home/ubuntu/DS/PKLot/tensorflow_ds/training/model.ckpt-4210 \
--output_directory=/home/ubuntu/DS/PKLot/tensorflow_ds/inference

