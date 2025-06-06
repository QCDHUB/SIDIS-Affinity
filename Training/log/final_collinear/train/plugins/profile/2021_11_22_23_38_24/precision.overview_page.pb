�	t��%:@t��%:@!t��%:@	W�(���?W�(���?!W�(���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLt��%:@�gy��5@1-	PS���?A	�/����?I4J����?Yk��=]��?rEagerKernelExecute 0*	Zd;ߟg@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Q�|�?!��<'V:@)�Q�|�?1��<'V:@:Preprocessing2U
Iterator::Model::ParallelMapV2��
��?!�,m���6@)��
��?1�,m���6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��C6��?!;�BQ7@)l��F���?1���y�5@:Preprocessing2F
Iterator::Model�b+hZb�?!?���A@)$(~��k�?1{0�8E*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateLm����?!� �N��@@)*�t�?1Hf"���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���N���?!a�>}tP@)0L�
F%�?19m�E1�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap4��k��?!��Vr�A@)�����n?1���o$��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV���4i?!�0>��?)V���4i?1�0>��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 83.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9W�(���?IL��s��V@Q����ę@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�gy��5@�gy��5@!�gy��5@      ��!       "	-	PS���?-	PS���?!-	PS���?*      ��!       2		�/����?	�/����?!	�/����?:	4J����?4J����?!4J����?B      ��!       J	k��=]��?k��=]��?!k��=]��?R      ��!       Z	k��=]��?k��=]��?!k��=]��?b      ��!       JGPUYW�(���?b qL��s��V@y����ę@�"B
&gradient_tape/model_1/dense_4/MatMul_1MatMul.����?!.����?"4
model_1/dense_4/MatMulMatMul�Q�?��?!�q���<�?0"B
$gradient_tape/model_1/dense_4/MatMulMatMul�[ks@��?!�� @�?0"D
&gradient_tape/model_1/dense_3/ReluGradReluGrad�e
V�?!���mU��?"4
model_1/dense_3/BiasAddBiasAdd���?!I���=�?".
model_1/dense_3/ReluRelu^���sY�?!�C�+��?"D
&gradient_tape/model_1/dense_4/ReluGradReluGrad�çk���?!?��}K��?"B
$gradient_tape/model_1/dense_3/MatMulMatMul����?!S�K�t�?0"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGradx5g��)�?!���+F�?".
model_1/dense_4/ReluRelux��[�֗?!K��.��?Q      Y@Y�18��5@a��18�S@q=Ze��;7@y�V/���?"�
both�Your program is POTENTIALLY input-bound because 83.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 