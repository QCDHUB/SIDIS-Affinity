	??ۻ?8@??ۻ?8@!??ۻ?8@	?(<
y???(<
y??!?(<
y??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??ۻ?8@y"??p6@1B?L?????A~T?~O???I]???!??Y????}r??rEagerKernelExecute 0*	?????k@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Ά?3???!8??d{6@)?Ά?3???18??d{6@:Preprocessing2U
Iterator::Model::ParallelMapV2????ѥ?!?4?jU?3@)????ѥ?1?4?jU?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}"O?????!?EX???2@)???h????10???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate>]ݱ?&??!pb??C@)??3g}ʡ?1?I_?0@:Preprocessing2F
Iterator::Modell?<*??!??ޒ\@@)~8H????1?vs#*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD6?.6???!0??6??P@)~q?J[\??1V?¸o!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~NA~6??!??cID@)??X?p?16C6<????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??E??\j?!???۽??)??E??\j?1???۽??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?(<
y??I??n?W@Q??8?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y"??p6@y"??p6@!y"??p6@      ??!       "	B?L?????B?L?????!B?L?????*      ??!       2	~T?~O???~T?~O???!~T?~O???:	]???!??]???!??!]???!??B      ??!       J	????}r??????}r??!????}r??R      ??!       Z	????}r??????}r??!????}r??b      ??!       JGPUY?(<
y??b q??n?W@y??8?@