	???WI@???WI@!???WI@	#u+??"@#u+??"@!#u+??"@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???WI@?Y?@1?H?[???Az?蹅??I???h?@Y?T[r??rEagerKernelExecute 0*2?Z,h@)       =2U
Iterator::Model::ParallelMapV2UMu???!ad4?@@)UMu???1ad4?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??N??!S????=5@)D????o??1o??t?3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????#ӡ?!??J-? 2@)????#ӡ?1??J-? 2@:Preprocessing2F
Iterator::Model?EИIԷ?!H???H@)?????P??1???]k?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?l<?b???!v?p??7@)&?fe????18 ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?аu???!?p*??I@)㊋?r??1?(?Z<?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd??u??!???<??9@)??z2??k?1R~??,0??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ù?i?!=???????)??ù?i?1=???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?33.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t48.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9$u+??"@ITg?TˬT@Q?
G&@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Y?@?Y?@!?Y?@      ??!       "	?H?[????H?[???!?H?[???*      ??!       2	z?蹅??z?蹅??!z?蹅??:	???h?@???h?@!???h?@B      ??!       J	?T[r???T[r??!?T[r??R      ??!       Z	?T[r???T[r??!?T[r??b      ??!       JGPUY$u+??"@b qTg?TˬT@y?
G&@