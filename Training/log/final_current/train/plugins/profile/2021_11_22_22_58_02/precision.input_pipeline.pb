	A?]??]<@A?]??]<@!A?]??]<@	t?y7ΰ??t?y7ΰ??!t?y7ΰ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLA?]??]<@?V'g(?7@1kdWZFj??A??????I??n? @Y4?fI??rEagerKernelExecute 0*	V-???h@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq???????!`??o?9@)q???????1`??o?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC˺,D??!???`??6@) !????1%'
x*D5@:Preprocessing2U
Iterator::Model::ParallelMapV2???ip??!????? 3@)???ip??1????? 3@:Preprocessing2F
Iterator::Model(*?T??!?h	PY?@@)???x??1??/???,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatet?Lh?X??!??UoB@)??Χ???1́??]6%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDkE?????!?K?WӗP@)S@?? k??1?$?H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??Pj??!Q6Pf?C@)??S??q?1?????? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%???wj?!?e??~??)%???wj?1?e??~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9t?y7ΰ??I+?? ̾W@Q??(??
@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V'g(?7@?V'g(?7@!?V'g(?7@      ??!       "	kdWZFj??kdWZFj??!kdWZFj??*      ??!       2	????????????!??????:	??n? @??n? @!??n? @B      ??!       J	4?fI??4?fI??!4?fI??R      ??!       Z	4?fI??4?fI??!4?fI??b      ??!       JGPUYt?y7ΰ??b q+?? ̾W@y??(??
@