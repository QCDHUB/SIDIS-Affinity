	}w+{@}w+{@!}w+{@	0Z??[ @0Z??[ @!0Z??[ @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL}w+{@פ???	@1??;?????A6X8I?Ǥ?Ie??E?@Y??++MJ??rEagerKernelExecute 0*	sh??|?r@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet?p??[??!4?%>8I@)t?p??[??14?%>8I@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??	?_???!??????-@)D??]L3??1pwC?~?+@:Preprocessing2U
Iterator::Model::ParallelMapV22?CP5??!e?????'@)2?CP5??1e?????'@:Preprocessing2F
Iterator::ModelZ??mē??!?,k?#?6@)?B ?8???1??2ܮ&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN??}????!P?0?\?L@)׆?q?&??1?q??@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZippA?,_???!?4?wFS@)?@j'??1x??n?)@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6ɏ?k??!dL?H?4M@)???]/Mq?1~B-0???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorH?3?9Ak?!?1??????)H?3?9Ak?1?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?37.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t43.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.90Z??[ @IV?+?->T@Q9-H???%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	פ???	@פ???	@!פ???	@      ??!       "	??;???????;?????!??;?????*      ??!       2	6X8I?Ǥ?6X8I?Ǥ?!6X8I?Ǥ?:	e??E?@e??E?@!e??E?@B      ??!       J	??++MJ????++MJ??!??++MJ??R      ??!       Z	??++MJ????++MJ??!??++MJ??b      ??!       JGPUY0Z??[ @b qV?+?->T@y9-H???%@