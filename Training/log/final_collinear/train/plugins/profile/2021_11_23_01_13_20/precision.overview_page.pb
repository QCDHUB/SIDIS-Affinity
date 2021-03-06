?	???D;@???D;@!???D;@	???A&?????A&??!???A&??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???D;@??Z}u?6@1;?ʃ????A???S??IB?/h!a@Y3??J&??rEagerKernelExecute 0*	????x?i@2U
Iterator::Model::ParallelMapV2??1˞??!?Ⓢ??@)??1˞??1?Ⓢ??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQ?\?mO??!ܾ?=/7@)Q?\?mO??1ܾ?=/7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?^???F??!+?q?XV3@)1?t?????1[??*?2@:Preprocessing2F
Iterator::Model'?y?3??!????H@)a???)??1;N?^0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea?9???!??j??;@)C???|͂?1wk????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf??@?9??!g?g<1?I@)t??gy|?1
#_S"?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????`???!@x?m?==@)?d??7ij?1s??= 0??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??.?d?!?*W???)??.?d?1?*W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???A&??I?3??`?V@Qܠtpf?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Z}u?6@??Z}u?6@!??Z}u?6@      ??!       "	;?ʃ????;?ʃ????!;?ʃ????*      ??!       2	???S?????S??!???S??:	B?/h!a@B?/h!a@!B?/h!a@B      ??!       J	3??J&??3??J&??!3??J&??R      ??!       Z	3??J&??3??J&??!3??J&??b      ??!       JGPUY???A&??b q?3??`?V@yܠtpf?@?"B
&gradient_tape/model_1/dense_4/MatMul_1MatMul#|?? ???!#|?? ???"4
model_1/dense_4/MatMulMatMulG?#D???!?a?^2P??0"B
$gradient_tape/model_1/dense_4/MatMulMatMul???????!ӯ?_?!??0"D
&gradient_tape/model_1/dense_3/ReluGradReluGradd向:	??!)??q??"4
model_1/dense_3/BiasAddBiasAdd_OBj???!߯㬷M??".
model_1/dense_3/ReluRelun????i??!????R???"D
&gradient_tape/model_1/dense_4/ReluGradReluGradǕK4???!??E???"B
$gradient_tape/model_1/dense_3/MatMulMatMul?ĝj)??!?;?N????0"R
1gradient_tape/model_1/dense_3/BiasAdd/BiasAddGradBiasAddGrad??e\?>??!?izA?Y??".
model_1/dense_4/ReluRelu???????!>?????Q      Y@Y?18??5@a??18?S@q????2@y? ??????"?
both?Your program is POTENTIALLY input-bound because 83.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 