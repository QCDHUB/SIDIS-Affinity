	?h??i?:@?h??i?:@!?h??i?:@	????c??????c??!????c??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?h??i?:@}?;l"?6@1?2?}ƅ??A=?ЕT??I????u???YJ{?/L???rEagerKernelExecute 0*	?|?5^?f@2U
Iterator::Model::ParallelMapV2?o%;6??!@?qHB@)?o%;6??1@?qHB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ٕ???! ?6&??2@)/???0??1??f4??1@:Preprocessing2F
Iterator::ModelLP÷?n??!??+PCJ@)A?w?鱝?1T9??y?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?&????!P?=??+@)?&????1P?=??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0???"??!k&????G@)??X32ȍ?1????? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?%Tpx??!?bLh??2@)!?Ky ??1Q??&'|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????!??!??o??4@)?+=)?j?1??[u????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?K??T?h?!]????)?K??T?h?1]????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????c??I?E????V@QM_F?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	}?;l"?6@}?;l"?6@!}?;l"?6@      ??!       "	?2?}ƅ???2?}ƅ??!?2?}ƅ??*      ??!       2	=?ЕT??=?ЕT??!=?ЕT??:	????u???????u???!????u???B      ??!       J	J{?/L???J{?/L???!J{?/L???R      ??!       Z	J{?/L???J{?/L???!J{?/L???b      ??!       JGPUY????c??b q?E????V@yM_F?@