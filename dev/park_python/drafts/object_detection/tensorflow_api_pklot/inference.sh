python $HOME/tensorflow_models/research/object_detection/inference/infer_detections.py \
--input_tfrecord_paths=$HOME/DS/PKLot/tensorflow_ds/val.record \
--output_tfrecord_path=$HOME/DS/PKLot/tensorflow_ds/output/inference_output.tfrecord \
--inference_graph=$HOME/DS/PKLot/tensorflow_ds/inference/frozen_inference_graph.pb \
--discard_image_pixels
