python $HOME/tensorflow_models/research/object_detection/inference/infer_detections.py \
--input_tfrecord_paths=$HOME/DS/cars/val.record \
--output_tfrecord_path=$HOME/DS/cars/inference_output.tfrecord \
--inference_graph=$HOME/DS/cars/training/frozen_inference_graph.pb \
--discard_image_pixels
