	?x??=@?x??=@!?x??=@	?t???t??!?t??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?x??=@????SG8@1ol?`q??AmU?Y??I?};??@Y???0`???rEagerKernelExecute 0*	??v??j@2U
Iterator::Model::ParallelMapV2??y7??!ؼ??#	A@)??y7??1ؼ??#	A@:Preprocessing2F
Iterator::Modelu???!?#??+L@)????ϧ?1???H"E6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?g\W̠?!?ӈ??k/@)?g\W̠?1?ӈ??k/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b??Ր??!??[ϓ?.@)0e?????1Q??8??,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Z???!a?F??7@)???'???1}?	|??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?8GW??!1?? K?E@)M?^?iN~?1??%X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq:?V?S??!??yG?8@)vk???i?1J?z?E??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??lXSYd?!/?T???)??lXSYd?1/?T???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?t??I??B?[W@Q3I,VA-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????SG8@????SG8@!????SG8@      ??!       "	ol?`q??ol?`q??!ol?`q??*      ??!       2	mU?Y??mU?Y??!mU?Y??:	?};??@?};??@!?};??@B      ??!       J	???0`??????0`???!???0`???R      ??!       Z	???0`??????0`???!???0`???b      ??!       JGPUY?t??b q??B?[W@y3I,VA-@