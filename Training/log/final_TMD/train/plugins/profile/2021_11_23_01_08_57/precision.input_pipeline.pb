	?*?) @?*?) @!?*?) @	ΗsM@ΗsM@!ΗsM@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?*?) @uw???@1?%:?,???A?x?&1??I?;??b?	@YΎT??E??rEagerKernelExecute 0*	"??~j?l@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ݰmQ??!V~????B@)??ݰmQ??1V~????B@:Preprocessing2U
Iterator::Model::ParallelMapV2??ο]???!???8??7@)??ο]???1???8??7@:Preprocessing2F
Iterator::ModelpA?,_??!~????C@)?(???Ǣ?1	Θ???/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat#?dTƝ?!?=?\+3)@)???????1QX??pm'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'L?????!L?k1??E@)MI???*??1?O?ү@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!?W}?7N@)Z???аx?1???#??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1?闈???!\ػ?ҜF@)BA)Z?h?1	J8d??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT????`?!YN`?[??)T????`?1YN`?[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t43.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??ΗsM@I??s?:U@Q1|xq?#@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	uw???@uw???@!uw???@      ??!       "	?%:?,????%:?,???!?%:?,???*      ??!       2	?x?&1???x?&1??!?x?&1??:	?;??b?	@?;??b?	@!?;??b?	@B      ??!       J	ΎT??E??ΎT??E??!ΎT??E??R      ??!       Z	ΎT??E??ΎT??E??!ΎT??E??b      ??!       JGPUY??ΗsM@b q??s?:U@y1|xq?#@