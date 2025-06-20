�	A�]��]<@A�]��]<@!A�]��]<@	t�y7ΰ�?t�y7ΰ�?!t�y7ΰ�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLA�]��]<@�V'g(�7@1kdWZFj�?A�����?I��n� @Y4�fI�?rEagerKernelExecute 0*	V-���h@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq�����?!`��o�9@)q�����?1`��o�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC˺,D�?!���`��6@) !���?1%'
x*D5@:Preprocessing2U
Iterator::Model::ParallelMapV2�ip�?!����� 3@)�ip�?1����� 3@:Preprocessing2F
Iterator::Model(*�T�?!�h	PY�@@)���x�?1�/���,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatet�Lh�X�?!��UoB@)��Χ��?1́��]6%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDkE����?!�K�WӗP@)S@�� k�?1�$�H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��Pj�?!Q6Pf�C@)��S��q?1������ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%���wj?!�e��~�?)%���wj?1�e��~�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9t�y7ΰ�?I+�� ̾W@Q�(��
@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�V'g(�7@�V'g(�7@!�V'g(�7@      ��!       "	kdWZFj�?kdWZFj�?!kdWZFj�?*      ��!       2	�����?�����?!�����?:	��n� @��n� @!��n� @B      ��!       J	4�fI�?4�fI�?!4�fI�?R      ��!       Z	4�fI�?4�fI�?!4�fI�?b      ��!       JGPUYt�y7ΰ�?b q+�� ̾W@y�(��
@�"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad�h�~���?!�h�~���?"B
$gradient_tape/model_2/dense_7/MatMulMatMul��4���?!���Y59�?0"4
model_2/dense_7/MatMulMatMul;�[���?!�qv����?0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMul����@>�?!&�l��?"4
model_2/dense_6/BiasAddBiasAdd�_E���?!BtL�?".
model_2/dense_6/ReluRelu�+�;�y�?!}�y��?"B
$gradient_tape/model_2/dense_6/MatMulMatMul2�*G9�?!��lru��?0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGrad��J;�t�?!�<!V�q�?"-
IteratorGetNext/_1_Send.H�5O��?!��Ii��?"4
model_2/dense_6/MatMulMatMulgyF��y�?!���#�?0Q      Y@Y�18��5@a��18�S@qҘ���8@yC5���?"�
both�Your program is POTENTIALLY input-bound because 84.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 