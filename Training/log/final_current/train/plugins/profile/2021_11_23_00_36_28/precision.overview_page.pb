?	?=Զ<@?=Զ<@!?=Զ<@	?:D????:D???!?:D???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?=Զ<@d?3?%?7@1XU/??d??A:x&4I,??I|?ԗ?=	@Y<0???D??rEagerKernelExecute 0*	?O??n?g@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice|`?? ??!	?VS?	9@)|`?? ??1	?VS?	9@:Preprocessing2U
Iterator::Model::ParallelMapV2O?P??&??!?~
??8@)O?P??&??1?~
??8@:Preprocessing2F
Iterator::Model?|?X????!<Y??[?E@)s+??X¢?1?3v??w3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?fF?N??!`lVi??1@)e?`TR'??1?ԙ???0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatecD?в???!?7=?ˑ@@)??kC?8??1?GFe3 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ?({K9??!Ħ?F?@L@)x?7N
?~?1????6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap衶???!???@?CA@)f?ʉve?1F9/?(F??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~?[?~lb?!w?;???)~?[?~lb?1w?;???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?:D???I@??.??W@Q)5?&	?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d?3?%?7@d?3?%?7@!d?3?%?7@      ??!       "	XU/??d??XU/??d??!XU/??d??*      ??!       2	:x&4I,??:x&4I,??!:x&4I,??:	|?ԗ?=	@|?ԗ?=	@!|?ԗ?=	@B      ??!       J	<0???D??<0???D??!<0???D??R      ??!       Z	<0???D??<0???D??!<0???D??b      ??!       JGPUY?:D???b q@??.??W@y)5?&	?@?"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad????E???!????E???"B
$gradient_tape/model_2/dense_7/MatMulMatMulz???ˑ??!?Jh?$??0"4
model_2/dense_7/MatMulMatMul??PՑ??!+T?y???0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMul????}???!1??C???"4
model_2/dense_6/BiasAddBiasAdd?U]Z????!?????".
model_2/dense_6/ReluRelu s?Qm??!<,ȇ???"B
$gradient_tape/model_2/dense_6/MatMulMatMul+Dku??!?h????0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGrad<*w??H??!#S z.$??"4
model_2/dense_6/MatMulMatMul^??u???!?B˕??0"-
IteratorGetNext/_1_Send0I梿??!??}?????Q      Y@Y?18??5@a??18?S@q?t?6@y,Dku??"?
both?Your program is POTENTIALLY input-bound because 84.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 