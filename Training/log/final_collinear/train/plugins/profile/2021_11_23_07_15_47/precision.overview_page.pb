?	?_w???@?_w???@!?_w???@	뾫vٔ@뾫vٔ@!뾫vٔ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?_w???@????@1ӆ???O??A<f?2?}??I?X?v? @Yū?m???rEagerKernelExecute 0*	?z?G%m@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?}8H????!'?lhbB@)s-Z??մ?1????sA@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??7j????!?ym??a7@)??7j????1?ym??a7@:Preprocessing2U
Iterator::Model::ParallelMapV2??.??[??!??B?=?'@)??.??[??1??B?=?'@:Preprocessing2F
Iterator::Model?R]????!?\X ??6@)?!??ƽ??1??mJ??%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?^D?1u??!???{?L@@)?"?????1I?(?$o"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip]ݱ?&??!?????US@)?Xl??Ɗ?1?nɳ?m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????޴?!???,?{A@)#-??#?v?1??y{?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-?}?q?!"p-?????)-?}?q?1"p-?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?33.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9꾫vٔ@I~@?*?P@Q*?͘?\=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	ӆ???O??ӆ???O??!ӆ???O??*      ??!       2	<f?2?}??<f?2?}??!<f?2?}??:	?X?v? @?X?v? @!?X?v? @B      ??!       J	ū?m???ū?m???!ū?m???R      ??!       Z	ū?m???ū?m???!ū?m???b      ??!       JGPUY꾫vٔ@b q~@?*?P@y*?͘?\=@?"@
$gradient_tape/model/dense_1/MatMul_1MatMul?r?"???!?r?"???"2
model/dense_1/MatMulMatMul ?0?:??!?5?????0"@
"gradient_tape/model/dense_1/MatMulMatMul?'?????!??{?T???0"@
"gradient_tape/model/dense/ReluGradReluGrad???
???!^?$????"0
model/dense/BiasAddBiasAdd-?0*֦??!???t???"*
model/dense/ReluRelu??u?z]??!?1߽K>??"B
$gradient_tape/model/dense_1/ReluGradReluGrad?F?iY??!GF?V?C??">
 gradient_tape/model/dense/MatMulMatMul?J'Ԯ??!????X)??0"N
-gradient_tape/model/dense/BiasAdd/BiasAddGradBiasAddGradx????Й?!Qh7U????",
model/dense_1/ReluRelus?????!A?I???Q      Y@Y?18??5@a??18?S@q????bU@yx????"?
both?Your program is POTENTIALLY input-bound because 34.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?85.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 