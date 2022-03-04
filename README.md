# Reference Software for Tiny Yolo v3 Quantization

* How to run?
To evaluate the original floating point model, run `tiny-yolo-aix2022.sh`.
To evaluate the quantized model, run `tiny-yolo-aix2022-int8.sh`.
You can also test the quantized model using `tiny-yolo-aix2022-int8-test.sh` and visualize the prediction result.

* Tips
To save the quantized model, use flag `-save_params` at the end of command (see `tiny-yolo-aix2022-int8.sh`)