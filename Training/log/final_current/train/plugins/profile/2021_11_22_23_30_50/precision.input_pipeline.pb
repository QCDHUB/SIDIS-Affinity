	?V????:@?V????:@!?V????:@	???@???@!???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?V????:@zq?.7@1__?R#t??AO??C??IP?eo?@Y+??p????rEagerKernelExecute 0*	䥛? &p@2U
Iterator::Model::ParallelMapV2Ͻ?K?;??!f?n?j:@)Ͻ?K?;??1f?n?j:@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezrM??Ϊ?!?~?R?C4@)zrM??Ϊ?1?~?R?C4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???P?v??!??Yf~2@)????????1L?)1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ڧ?1??!bh??w?A@)??g?????1	??D??-@:Preprocessing2F
Iterator::Model?KqU?w??!?p?_@C@)???x??18s?.??(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?,??V??!???P??N@))??????1BhA?d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?U?3???!?A??
tC@)?^Ӄ?R??1+?M?/?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?٭e2l?!S??hP??)?٭e2l?1S??hP??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???@IE?G=?W@Q?߹L8Y@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	zq?.7@zq?.7@!zq?.7@      ??!       "	__?R#t??__?R#t??!__?R#t??*      ??!       2	O??C??O??C??!O??C??:	P?eo?@P?eo?@!P?eo?@B      ??!       J	+??p????+??p????!+??p????R      ??!       Z	+??p????+??p????!+??p????b      ??!       JGPUY???@b qE?G=?W@y?߹L8Y@