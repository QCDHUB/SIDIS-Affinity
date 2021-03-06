?	????4?<@????4?<@!????4?<@	,?D?? @,?D?? @!,?D?? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????4?<@xρ?98@14H?S???A@?t?_???IF#?W<?@Y4?i??r??rEagerKernelExecute 0*	???x??k@2U
Iterator::Model::ParallelMapV2?1Xq????!??J@F=@)?1Xq????1??J@F=@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicez?3M?~??!?B(?57@)z?3M?~??1?B(?57@:Preprocessing2F
Iterator::Model?&c`??!??w?G@):?}?kϤ?1?*aB?:2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(??h????!????$2@)Ll>???1?^?{.?0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??[????!>ih9??J@)???[???1`??ȝH@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea??????!???v?;@)????}???1|???u?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'??2???!?=?;Đ=@)???q?jm?1%Sc?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?-X?xi?!?Zx?dO??)?-X?xi?1?Zx?dO??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9+?D?? @I?a?%o?W@Q?ώM ?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	xρ?98@xρ?98@!xρ?98@      ??!       "	4H?S???4H?S???!4H?S???*      ??!       2	@?t?_???@?t?_???!@?t?_???:	F#?W<?@F#?W<?@!F#?W<?@B      ??!       J	4?i??r??4?i??r??!4?i??r??R      ??!       Z	4?i??r??4?i??r??!4?i??r??b      ??!       JGPUY+?D?? @b q?a?%o?W@y?ώM ?@?"D
&gradient_tape/model_2/dense_6/ReluGradReluGrad/??轆??!/??轆??"B
$gradient_tape/model_2/dense_7/MatMulMatMul?8??i??!?mX?\???0"4
model_2/dense_7/MatMulMatMuldFRQ>$??!`? >???0"B
&gradient_tape/model_2/dense_7/MatMul_1MatMul??pӒ??!b,#??i??"4
model_2/dense_6/BiasAddBiasAdd??]ݕ??!???Bj???".
model_2/dense_6/ReluRelu???:M??!^R?d\???"B
$gradient_tape/model_2/dense_6/MatMulMatMul???"o???!Ț?V?`??0"R
1gradient_tape/model_2/dense_6/BiasAdd/BiasAddGradBiasAddGrad2???ò??!??ǐ??"4
model_2/dense_6/MatMulMatMulf[}r??!Qۗ?V\??0"-
IteratorGetNext/_1_SendJ?%????!?5
???Q      Y@Y?18??5@a??18?S@q??y?\?=@y??f?'M??"?
both?Your program is POTENTIALLY input-bound because 83.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?29.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 