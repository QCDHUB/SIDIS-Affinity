	*7QKsK@*7QKsK@!*7QKsK@	???х?@???х?@!???х?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL*7QKsK@?????t@1ADj?Ŵ??AB??	ܲ?I??.5B?@Y???P???rEagerKernelExecute 0*	I+?jo@2Z
#Iterator::Model::ParallelMapV2::Zip*???P??!??????O@)?lw?N??1^??B??:@:Preprocessing2U
Iterator::Model::ParallelMapV2r 
fL??!?9l??3@)r 
fL??1?9l??3@:Preprocessing2F
Iterator::Model@???൷?!zcmB@)???[??1*?Z11@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatc ?={??!%jry?x1@)5`??i??1zĄ?nb0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Q?????!?i)??+@)?Q?????1?i)??+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??8??¦?!??;?8?1@)??֪]??1??6?d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??k?)??!U??}6?2@)?O?mpf?1m?r?o??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????9]f?!?Z?.a??)????9]f?1?Z?.a??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?48.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t33.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???х?@I?Ο??T@QhI??%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????t@?????t@!?????t@      ??!       "	ADj?Ŵ??ADj?Ŵ??!ADj?Ŵ??*      ??!       2	B??	ܲ?B??	ܲ?!B??	ܲ?:	??.5B?@??.5B?@!??.5B?@B      ??!       J	???P??????P???!???P???R      ??!       Z	???P??????P???!???P???b      ??!       JGPUY???х?@b q?Ο??T@yhI??%@