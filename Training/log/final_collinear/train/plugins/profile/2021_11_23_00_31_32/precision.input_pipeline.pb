	/R(_�=@/R(_�=@!/R(_�=@	�H�pMK�?�H�pMK�?!�H�pMK�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL/R(_�=@Ƌ�!rB9@1�̒ 5��?AZ��c!:�?IC8�&@Y�=yX��?rEagerKernelExecute 0*	�l���af@2U
Iterator::Model::ParallelMapV2X Sh�?!�Qt��;@)X Sh�?1�Qt��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���GS=�?!�saP�;@)v�[����?1p%���8@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�9���?!�;�qY83@)�9���?1�;�qY83@:Preprocessing2F
Iterator::Model�0�q�	�?!`glo?�D@)���}V�?1}SՎ�+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�[Y��̺?!�����;M@)a��pɁ?1"m��f@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�۞ �ݥ?!�x��
�7@)�+I����?1"��/Ɔ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorE�N���t?!��A��@)E�N���t?1��A��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa����?!E�Ik�:@)>���4`p?1eK�j��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�H�pMK�?I���<W@QЪ��	I@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ƌ�!rB9@Ƌ�!rB9@!Ƌ�!rB9@      ��!       "	�̒ 5��?�̒ 5��?!�̒ 5��?*      ��!       2	Z��c!:�?Z��c!:�?!Z��c!:�?:	C8�&@C8�&@!C8�&@B      ��!       J	�=yX��?�=yX��?!�=yX��?R      ��!       Z	�=yX��?�=yX��?!�=yX��?b      ��!       JGPUY�H�pMK�?b q���<W@yЪ��	I@