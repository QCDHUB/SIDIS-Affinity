	�xy�@�xy�@!�xy�@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�xy�@&ǝ���@1� �4�?AG�&ji�?I�'�XQ�@rEagerKernelExecute 0*	-���l@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�:�zj�?!d��9@)�:�zj�?1d��9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��ͪ?!�͋�c�6@)�u��$�?1�	/�@5@:Preprocessing2F
Iterator::Model}	^�?!�/�'�A@)F'K���?1fѸ�2@:Preprocessing2U
Iterator::Model::ParallelMapV2��ފ��?!!�%aB�1@)��ފ��?1!�%aB�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�l\���?!�D�g��@@)=�u��?1�*!ڍ� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�4bf���?!hplP@)��c�M*�?1\�s�P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl>��?!�=�f��A@)��"[As?1���?l @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU���)n?!*�#hڹ�?)U���)n?1*�#hڹ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 30.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�51.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID^�ʳ�T@Q�^�0I1@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	&ǝ���@&ǝ���@!&ǝ���@      ��!       "	� �4�?� �4�?!� �4�?*      ��!       2	G�&ji�?G�&ji�?!G�&ji�?:	�'�XQ�@�'�XQ�@!�'�XQ�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qD^�ʳ�T@y�^�0I1@