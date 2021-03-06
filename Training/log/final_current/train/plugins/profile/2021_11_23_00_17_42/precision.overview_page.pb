?	*?"<@*?"<@!*?"<@	??n?2????n?2??!??n?2??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL*?"<@?1?3/?7@1Ǹ???\??A?b??*3??I?+?F<@YiƢ????rEagerKernelExecute 0*	h??|?af@2U
Iterator::Model::ParallelMapV21}?!8.??!??H??=@)1}?!8.??1??H??=@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceiR
?????!?82?H?3@)iR
?????1?82?H?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6W?sD???!H?2 ??5@)s?4?B??1?z?r?1@:Preprocessing2F
Iterator::ModelG?	1?T??!n1?`?DG@)?:ǀ????1Z?x+?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?p?Qe??!/?s?19@)?x$^?΅?1$1??
?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ի?耸?!??5??J@)????u???1?̐unD@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??i?{?!?2Sk?d@)??i?{?1?2Sk?d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps??/٨?!??Ԡ`;@)()? ?l?1?7??R???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??n?2??I%K????W@Q|?Z?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1?3/?7@?1?3/?7@!?1?3/?7@      ??!       "	Ǹ???\??Ǹ???\??!Ǹ???\??*      ??!       2	?b??*3???b??*3??!?b??*3??:	?+?F<@?+?F<@!?+?F<@B      ??!       J	iƢ????iƢ????!iƢ????R      ??!       Z	iƢ????iƢ????!iƢ????b      ??!       JGPUY??n?2??b q%K????W@y|?Z?@?"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad???????!???????"B
$gradient_tape/model_2/dense_7/MatMulMatMul$j?z??!d?Y?W/??0"4
model_2/dense_7/MatMulMatMul_???3??!J??????0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMulrLk????!f?ޔ???"4
model_2/dense_6/BiasAddBiasAddmYB????!a?7? ??".
model_2/dense_6/ReluRelu?[s?????!?$?????"B
$gradient_tape/model_2/dense_6/MatMulMatMulٻ\0?
??!????E???0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGrad?4??ê?!??y?5??"4
model_2/dense_6/MatMulMatMul?_?k???!?Yi ????0"-
IteratorGetNext/_1_Send??|?-???!?&?D???Q      Y@Y?18??5@a??18?S@q?鉋:@y?'?0&y??"?
both?Your program is POTENTIALLY input-bound because 84.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?26.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 