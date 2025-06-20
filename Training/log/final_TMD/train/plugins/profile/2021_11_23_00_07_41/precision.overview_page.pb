�	�W��*@�W��*@!�W��*@	w��X@w��X@!w��X@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�W��*@���/J�@1̵h���?A����);�?I��oa�8�?Y������?rEagerKernelExecute 0*	�rh��n@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat'�Ҩ��?!��g<@)�A��v��?1l�e2+�:@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��%��:�?!������4@)��%��:�?1������4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatelC�8�?!v�N��D@)�d�,�?1��B��4@:Preprocessing2U
Iterator::Model::ParallelMapV2΋_�(�?!O�^Fg (@)΋_�(�?1O�^Fg (@:Preprocessing2F
Iterator::Model&7��5�?!#z���6@)@�Z�kB�?1�)�X�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?9
3�?!xa��[S@)�.6��?1�x�Pb@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��Q���?!t'��F@)�����w?1��m�z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���!�q?!�ubx5��?)���!�q?1�ubx5��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 49.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9w��X@I�"��pT@Q����;�*@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���/J�@���/J�@!���/J�@      ��!       "	̵h���?̵h���?!̵h���?*      ��!       2	����);�?����);�?!����);�?:	��oa�8�?��oa�8�?!��oa�8�?B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?b      ��!       JGPUYw��X@b q�"��pT@y����;�*@�"2
model/dense_1/MatMulMatMul��<%�q�?!��<%�q�?0"@
$gradient_tape/model/dense_1/MatMul_1MatMul�D��?!�j��1�?"@
"gradient_tape/model/dense_1/MatMulMatMul���Or�?!e_�<X5�?0"@
"gradient_tape/model/dense/ReluGradReluGradCmq�W�?!��(�SK�?"0
model/dense/BiasAddBiasAdd��.L֭?!F��B�?"*
model/dense/ReluRelu��.L֭?!�9�bs`�?">
 gradient_tape/model/dense/MatMulMatMul#��뻧?!���2��?0"-
IteratorGetNext/_1_Sendn�X�j�?!��Q��R�?"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGradCmq�W�?!��8�Y��?"0
model/dense/MatMulMatMulDc%~1�?!����?0Q      Y@Y�18��5@a��18�S@q4�P�@y����m�?"�

both�Your program is POTENTIALLY input-bound because 49.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 