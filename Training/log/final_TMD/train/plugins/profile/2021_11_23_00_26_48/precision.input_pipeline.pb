	 o���@ o���@! o���@	���*v@���*v@!���*v@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL o���@�LLb�@1G�tF^�?AAaP���?I��1�q@YCr2q��?rEagerKernelExecute 0*	y�&1�h@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ECƣT�?!�ELt��9@)�G�?��?1���H�5@:Preprocessing2U
Iterator::Model::ParallelMapV2ꗈ�ο�?!����i5@)ꗈ�ο�?1����i5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceg��͢?!hL�zi�2@)g��͢?1hL�zi�2@:Preprocessing2F
Iterator::ModelI�L���?!����B@)����?1.�ĕٜ/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!�?R�]�=@)$}ZEh�?1*睵�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��B��?!�&�cO@)臭����?1G��O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���ْU�?!�аm@)���ْU�?1�аm@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���D�?!���ø@@)-��\n0t?1�;���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�34.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t46.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���*v@Ij�jc7ST@QPEO=�&@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�LLb�@�LLb�@!�LLb�@      ��!       "	G�tF^�?G�tF^�?!G�tF^�?*      ��!       2	AaP���?AaP���?!AaP���?:	��1�q@��1�q@!��1�q@B      ��!       J	Cr2q��?Cr2q��?!Cr2q��?R      ��!       Z	Cr2q��?Cr2q��?!Cr2q��?b      ��!       JGPUY���*v@b qj�jc7ST@yPEO=�&@