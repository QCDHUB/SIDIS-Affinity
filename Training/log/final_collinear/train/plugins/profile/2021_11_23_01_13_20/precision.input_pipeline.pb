	���D;@���D;@!���D;@	���A&�?���A&�?!���A&�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���D;@��Z}u�6@1;�ʃ���?A���S�?IB�/h!a@Y3��J&�?rEagerKernelExecute 0*	����x�i@2U
Iterator::Model::ParallelMapV2��1˞�?!�Ⓢ�?@)��1˞�?1�Ⓢ�?@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQ�\�mO�?!ܾ�=/7@)Q�\�mO�?1ܾ�=/7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�^���F�?!+�q�XV3@)1�t����?1[��*�2@:Preprocessing2F
Iterator::Model'�y�3�?!����H@)a���)�?1;N�^0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea�9��?!��j�;@)C���|͂?1wk���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf��@�9�?!g�g<1�I@)t��gy|?1
#_S"�
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����`��?!@x�m�==@)�d��7ij?1s�= 0�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��.�d?!�*W��?)��.�d?1�*W��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 83.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���A&�?I�3��`�V@Qܠtpf�@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Z}u�6@��Z}u�6@!��Z}u�6@      ��!       "	;�ʃ���?;�ʃ���?!;�ʃ���?*      ��!       2	���S�?���S�?!���S�?:	B�/h!a@B�/h!a@!B�/h!a@B      ��!       J	3��J&�?3��J&�?!3��J&�?R      ��!       Z	3��J&�?3��J&�?!3��J&�?b      ��!       JGPUY���A&�?b q�3��`�V@yܠtpf�@