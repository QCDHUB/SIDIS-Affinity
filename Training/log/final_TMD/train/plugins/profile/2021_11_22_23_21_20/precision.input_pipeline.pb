	???AB@???AB@!???AB@	]!|?!@]!|?!@!]!|?!@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???AB@?WV???@1???O???A$ӡ??n??I!???
@Y@??
???rEagerKernelExecute 0*	?? ?rdk@2U
Iterator::Model::ParallelMapV2ɏ?k???!T?_?W6@)ɏ?k???1T?_?W6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel??????!IKki??3@)l??????1IKki??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?ѩ+??!.?Q??4@)???ᱟ??1????E3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?b?0훳?!??O?zA@)<??~K??1O?h{?.@:Preprocessing2F
Iterator::Modelo??=δ?!???!?B@)?0????1d룡?.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL8????!}?d?tO@)???SV??18D?<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB!?J??!??????B@)???=?z?1l?\???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\p?h?!Ct?Ӧ??)?\p?h?1Ct?Ӧ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?47.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t31.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9]!|?!@I@?	-?T@Q????R&@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?WV???@?WV???@!?WV???@      ??!       "	???O??????O???!???O???*      ??!       2	$ӡ??n??$ӡ??n??!$ӡ??n??:	!???
@!???
@!!???
@B      ??!       J	@??
???@??
???!@??
???R      ??!       Z	@??
???@??
???!@??
???b      ??!       JGPUY]!|?!@b q@?	-?T@y????R&@